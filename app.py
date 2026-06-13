import os

# Отключаем многопоточность токенизатора и ограничиваем потоки PyTorch ДО импорта torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from catboost import CatBoostClassifier, CatBoostRegressor
from config import CONFIG

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the tokenizer and the model...")
    device = "cpu"
    torch.set_num_threads(1)

    ml_models["tokenizer"] = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    ml_models["bert_model"] = AutoModel.from_pretrained(CONFIG['model_name']).to(device)
    ml_models["bert_model"].eval()

    print("Loading CatBoost models...")
    ml_models["risk_model"] = CatBoostClassifier()
    ml_models["risk_model"].load_model(CONFIG['catboost_risk_model_save_path'])

    ml_models["est_model"] = CatBoostRegressor()
    ml_models["est_model"].load_model(CONFIG['catboost_estimate_model_save_path'])

    print("The tokenizer and the model have been loaded successfully")
    yield
    ml_models.clear()

app = FastAPI(title="Jira Estimate & Risk Predictor", lifespan=lifespan)

class JiraTask(BaseModel):
    summary: str
    description: str = ""

def get_embedding(text: str) -> np.ndarray:
    inputs = ml_models["tokenizer"](text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = ml_models["bert_model"](**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    return embedding

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": "tokenizer" in ml_models}

@app.post("/predict")
def predict(task: JiraTask):
    try:
        if "est_model" not in ml_models or "risk_model" not in ml_models:
            raise HTTPException(status_code=503, detail="Models aren't loaded yet")

        full_text = f"{task.summary} {task.description}".strip()
        if not full_text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        embedding = get_embedding(full_text).reshape(1, -1)

        pred_log = ml_models["est_model"].predict(embedding)[0]
        estimate_fte = max(float(np.expm1(pred_log)), 0.0)

        risk_class = int(ml_models["risk_model"].predict(embedding)[0])
        risk_label = "Has Risk" if risk_class == 1 else "No Risk"

        return {
            "estimate_fte": round(estimate_fte, 3),
            "estimate_hours": round(estimate_fte * 8, 1),
            "risk_level": risk_label
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
