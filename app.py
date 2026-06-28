import os
import pickle
from contextlib import asynccontextmanager

from catboost import CatBoostClassifier, CatBoostRegressor
from fastapi import FastAPI

from config import CONFIG
from feedback import init_db
from routes import router
from state import ml_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(CONFIG["feedback_db_path"])

    print("Loading fast LSA processing pipeline...")
    pipeline_path = os.path.join(CONFIG["embeddings_save_path"], "bm25_lsa_pipeline.pkl")
    with open(pipeline_path, "rb") as f:
        artifacts = pickle.load(f)
    ml_models["vectorizer"] = artifacts["vectorizer"]
    ml_models["svd_transformer"] = artifacts["svd_transformer"]

    print("Loading 5-Fold Regressors...")
    ml_models["est_folds"] = []
    for fold in range(5):
        path = os.path.join(
            os.path.dirname(CONFIG["catboost_estimate_model_save_path"]),
            f"estimate_fold_{fold}.cbm",
        )
        ml_models["est_folds"].append(CatBoostRegressor().load_model(path))

    print("Loading 5-Fold Classifiers...")
    ml_models["risk_folds"] = []
    for fold in range(5):
        path = os.path.join(
            os.path.dirname(CONFIG["catboost_risk_model_save_path"]),
            f"risk_fold_{fold}.cbm",
        )
        ml_models["risk_folds"].append(CatBoostClassifier().load_model(path))

    print("All models loaded.")
    yield
    ml_models.clear()


app = FastAPI(title="Jira Multi-Modal L-KFold Predictor", lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
