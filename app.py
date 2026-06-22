import os
import pickle
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from config import CONFIG

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading fast LSA processing pipeline...")
    pipeline_path = os.path.join(CONFIG['embeddings_save_path'], "bm25_lsa_pipeline.pkl")
    with open(pipeline_path, "rb") as f:
        artifacts = pickle.load(f)
    ml_models["vectorizer"] = artifacts["vectorizer"]
    ml_models["svd_transformer"] = artifacts["svd_transformer"]

    print("Loading 5-Fold Regressors...")
    ml_models["est_folds"] = []
    for fold in range(5):
        path = os.path.join(os.path.dirname(CONFIG['catboost_estimate_model_save_path']), f"estimate_fold_{fold}.cbm")
        model = CatBoostRegressor().load_model(path)
        ml_models["est_folds"].append(model)

    print("Loading 5-Fold Classifiers...")
    ml_models["risk_folds"] = []
    for fold in range(5):
        path = os.path.join(os.path.dirname(CONFIG['catboost_risk_model_save_path']), f"risk_fold_{fold}.cbm")
        model = CatBoostClassifier().load_model(path)
        ml_models["risk_folds"].append(model)

    print("All lightweight models loaded successfully into lifespan")
    yield
    ml_models.clear()


app = FastAPI(title="Jira Multi-Modal L-KFold Predictor", lifespan=lifespan)


class JiraTask(BaseModel):
    summary: str
    description: str = ""
    region: str = "Unknown"
    subsystem: str = "Unknown"
    commitments: str = "Unknown"


def extract_features(summary: str, description: str, region: str, subsystem: str, commitments: str) -> pd.DataFrame:
    full_text = f"{summary} {description}".strip()

    has_description = 1 if description.strip() else 0
    has_code_block = 1 if "{code" in description.lower() else 0

    text_lower = full_text.lower()
    is_dev_task = 1 if any(w in text_lower for w in ['разраб', 'dev', 'implement', 'feature', 'фич', 'кодиров']) else 0
    is_test_task = 1 if any(w in text_lower for w in ['тест', 'test', 'qa', 'проверк', 'autotest', 'автотест']) else 0
    is_analysis_task = 1 if any(
        w in text_lower for w in ['анализ', 'anali', 'тз', 'требован', 'проектир', 'requirement']) else 0

    text_len = len(full_text)
    word_count = len(full_text.split())

    # Generate fast LSA components using the pre-trained fast TF-IDF/SVD workflow
    text_sparse = ml_models["vectorizer"].transform([full_text])
    lsa_embedding = ml_models["svd_transformer"].transform(text_sparse).flatten()

    num_emb_features = len(lsa_embedding)
    emb_cols = [f'emb_{i}' for i in range(num_emb_features)]
    df = pd.DataFrame([lsa_embedding], columns=emb_cols)

    df['has_description'] = float(has_description)
    df['has_code_block'] = float(has_code_block)
    df['is_dev_task'] = float(is_dev_task)
    df['is_test_task'] = float(is_test_task)
    df['is_analysis_task'] = float(is_analysis_task)
    df['text_len'] = float(text_len)
    df['word_count'] = float(word_count)

    df['region'] = str(region)
    df['subsystem'] = str(subsystem)
    df['commitments'] = str(commitments)

    return df


@app.get("/health")
def health_check():
    return {"status": "ok", "folds_loaded": len(ml_models.get("est_folds", [])) == 5}


@app.post("/predict")
def predict(task: JiraTask):
    try:
        if not task.summary.strip():
            raise HTTPException(status_code=400, detail="Summary cannot be empty")

        feature_df = extract_features(
            task.summary, task.description,
            task.region, task.subsystem, task.commitments
        )

        cat_features = ['region', 'subsystem', 'commitments']
        pool = Pool(data=feature_df, cat_features=cat_features)

        reg_preds_log = [model.predict(pool) for model in ml_models["est_folds"]]
        mean_pred_log = np.mean(reg_preds_log)
        base_estimate_days = float(np.expm1(mean_pred_log))
        base_estimate_hours = max(0.0, base_estimate_days * 8)

        prob_list = [model.predict_proba(pool) for model in ml_models["risk_folds"]]
        mean_probabilities = np.mean(prob_list, axis=0).flatten()

        prob_low = float(mean_probabilities[0])
        prob_medium = float(mean_probabilities[1])
        prob_critical = float(mean_probabilities[2])

        risk_buffer_multiplier = 1.0 + (0.15 * prob_medium) + (0.50 * prob_critical)
        adjusted_estimate_hours = base_estimate_hours * risk_buffer_multiplier

        return {
            "predicted_time_hours": round(base_estimate_hours, 1),
            "adjusted_time_with_buffer_hours": round(adjusted_estimate_hours, 1),
            "risk_profile": {
                "low_risk_prob_pct": round(prob_low * 100, 1),
                "medium_risk_prob_pct": round(prob_medium * 100, 1),
                "critical_risk_prob_pct": round(prob_critical * 100, 1)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
