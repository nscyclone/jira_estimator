import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from catboost import CatBoostClassifier, CatBoostRegressor
from config import CONFIG

app = FastAPI(title="Jira Estimate & Risk Predictor")

device = "cpu"

print("Loading the tokenizer and the model...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
bert_model = AutoModel.from_pretrained(CONFIG['model_name']).to(device)
bert_model.eval()

print("Loading CatBoost models...")
risk_model = CatBoostClassifier()
risk_model.load_model(CONFIG['catboost_risk_model_save_path'])

est_model = CatBoostRegressor()
est_model.load_model(CONFIG['catboost_estimate_model_save_path'])

# Request
class JiraTask(BaseModel):
    summary: str
    description: str = ""


def get_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    return embedding

@app.post("/predict")
def predict(task: JiraTask):
    try:
        full_text = f"{task.summary} {task.description}".strip()
        if not full_text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        embedding = get_embedding(full_text).reshape(1, -1)

        pred_log = est_model.predict(embedding)[0]
        estimate_fte = max(float(np.expm1(pred_log)), 0.0)

        risk_class = int(risk_model.predict(embedding)[0])
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
