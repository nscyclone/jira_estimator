import os
import pickle
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from config import CONFIG
from feature_engineering import compute_text_features

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

    features = compute_text_features(full_text, description)

    text_sparse = ml_models["vectorizer"].transform([full_text])
    lsa_embedding = ml_models["svd_transformer"].transform(text_sparse).flatten()

    emb_cols = [f'emb_{i}' for i in range(len(lsa_embedding))]
    df = pd.DataFrame([lsa_embedding], columns=emb_cols)

    for key, val in features.items():
        df[key] = float(val)

    df['region'] = str(region)
    df['subsystem'] = str(subsystem)
    df['commitments'] = str(commitments)

    return df


@app.get("/health")
def health_check():
    return {"status": "ok", "folds_loaded": len(ml_models.get("est_folds", [])) == 5}


@app.post("/predict")
def predict(task: JiraTask):
    if not task.summary.strip():
        raise HTTPException(status_code=400, detail="Summary cannot be empty")

    try:
        feature_df = extract_features(
            task.summary, task.description,
            task.region, task.subsystem, task.commitments,
        )

        cat_features = ['region', 'subsystem', 'commitments']
        pool = Pool(data=feature_df, cat_features=cat_features)

        reg_preds_log = [model.predict(pool) for model in ml_models["est_folds"]]
        mean_pred_log = np.mean(reg_preds_log)
        base_estimate_days = float(np.expm1(mean_pred_log))
        base_estimate_hours = max(0.0, base_estimate_days * CONFIG['workday_hours'])

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
                "critical_risk_prob_pct": round(prob_critical * 100, 1),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
