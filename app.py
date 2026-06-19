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
        base_estimate = float(np.expm1(pred_log))

        risk_probs = ml_models["risk_model"].predict_proba(embedding)[0]
        prob_has_risk = risk_probs[1]

        # "Has Risk" means that time_spent >= 1.25 * estimate
        # When risk is predicted the estimate can be adjusted by a static coefficient
        risk_buffer_multiplier = 1.0 + (0.30 * prob_has_risk)
        adjusted_estimate = base_estimate * risk_buffer_multiplier

        return {
            "base_estimate_hours": round(base_estimate * 8, 1),
            "adjusted_estimate_hours": round(adjusted_estimate * 8, 1),
            "risk_probability_pct": round(prob_has_risk * 100, 1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
