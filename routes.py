import subprocess
import sys

import numpy as np
import pandas as pd
from catboost import Pool
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import CONFIG
from feature_engineering import compute_text_features
from feedback import (
    get_feedback_count,
    get_feedback_metrics,
    get_latest_model_run,
    get_unused_feedback_count,
    insert_feedback,
)
from state import ml_models

router = APIRouter()

CAT_FEATURES = ["region", "subsystem", "commitments"]


class JiraTask(BaseModel):
    summary: str
    description: str = ""
    region: str = "Unknown"
    subsystem: str = "Unknown"
    commitments: str = "Unknown"


class FeedbackRequest(BaseModel):
    summary: str
    description: str = ""
    region: str = "Unknown"
    subsystem: str = "Unknown"
    commitments: str = "Unknown"
    predicted_days: float
    actual_days: float


def _extract_features(
    summary: str, description: str, region: str, subsystem: str, commitments: str
) -> pd.DataFrame:
    full_text = f"{summary} {description}".strip()
    features = compute_text_features(full_text, description)
    text_sparse = ml_models["vectorizer"].transform([full_text])
    lsa = ml_models["svd_transformer"].transform(text_sparse).flatten()
    df = pd.DataFrame([lsa], columns=[f"emb_{i}" for i in range(len(lsa))])
    for key, val in features.items():
        df[key] = float(val)
    df["region"] = str(region)
    df["subsystem"] = str(subsystem)
    df["commitments"] = str(commitments)
    return df


@router.get("/health")
def health_check():
    est_run = get_latest_model_run(CONFIG["feedback_db_path"], "estimate")
    return {
        "status": "ok",
        "folds_loaded": len(ml_models.get("est_folds", [])) == 5,
        "model_revision": est_run["run_id"][:7] if est_run else None,
        "model_published_at": est_run["created_at"] if est_run else None,
    }


@router.post("/predict")
def predict(task: JiraTask):
    if not task.summary.strip():
        raise HTTPException(status_code=400, detail="Summary cannot be empty")
    try:
        feature_df = _extract_features(
            task.summary, task.description, task.region, task.subsystem, task.commitments
        )
        pool = Pool(data=feature_df, cat_features=CAT_FEATURES)

        reg_preds_log = [m.predict(pool) for m in ml_models["est_folds"]]
        base_days = float(np.expm1(np.mean(reg_preds_log)))
        preds_days = np.expm1(np.array(reg_preds_log).flatten())
        prediction_std_days = float(np.std(preds_days))
        base_hours = max(0.0, base_days * CONFIG["workday_hours"])

        prob_list = [m.predict_proba(pool) for m in ml_models["risk_folds"]]
        probs = np.mean(prob_list, axis=0).flatten()
        p_low, p_medium, p_critical = float(probs[0]), float(probs[1]), float(probs[2])

        buffer = 1.0 + 0.15 * p_medium + 0.50 * p_critical
        adjusted_hours = base_hours * buffer

        return {
            "predicted_time_hours": round(base_hours, 1),
            "adjusted_time_with_buffer_hours": round(adjusted_hours, 1),
            "prediction_std_days": round(prediction_std_days, 2),
            "risk_profile": {
                "low_risk_prob_pct": round(p_low * 100, 1),
                "medium_risk_prob_pct": round(p_medium * 100, 1),
                "critical_risk_prob_pct": round(p_critical * 100, 1),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
def post_feedback(req: FeedbackRequest):
    if not req.summary.strip():
        raise HTTPException(status_code=400, detail="Summary cannot be empty")
    insert_feedback(
        CONFIG["feedback_db_path"],
        summary=req.summary,
        description=req.description,
        region=req.region,
        subsystem=req.subsystem,
        commitments=req.commitments,
        predicted_days=req.predicted_days,
        actual_days=req.actual_days,
    )
    count = get_feedback_count(CONFIG["feedback_db_path"])
    unused = get_unused_feedback_count(CONFIG["feedback_db_path"])
    delta_pct = (
        (req.actual_days - req.predicted_days) / req.actual_days * 100
        if req.actual_days
        else None
    )
    return {
        "delta_pct": round(delta_pct, 1) if delta_pct is not None else None,
        "feedback_count": count,
        "unused_feedback_count": unused,
        "retrain_ready": unused >= CONFIG["retrain_threshold"],
    }


@router.get("/metrics")
def get_metrics():
    db = CONFIG["feedback_db_path"]
    est_run = get_latest_model_run(db, "estimate")
    risk_run = get_latest_model_run(db, "risk")
    fb = get_feedback_metrics(db)
    count = fb["feedbacks_collected"]
    return {
        "model_revision": est_run["run_id"][:7] if est_run else None,
        "model_published_at": est_run["created_at"] if est_run else None,
        "model_trigger": est_run["trigger"] if est_run else None,
        "model_r2": est_run["r2"] if est_run else None,
        "model_mae": est_run["mae"] if est_run else None,
        "model_overrun_recall": risk_run["overrun_recall"] if risk_run else None,
        "feedbacks_collected": count,
        "feedback_mean_delta_pct": fb["feedback_mean_delta_pct"],
        "feedback_accuracy_within_25pct": fb["feedback_accuracy_within_25pct"],
        "estimated_hours_saved_per_sprint": round(count * 15 / 60, 1),
    }


@router.post("/retrain")
def trigger_retrain():
    unused = get_unused_feedback_count(CONFIG["feedback_db_path"])
    threshold = CONFIG["retrain_threshold"]
    if unused < threshold:
        return {
            "status": "skipped",
            "unused_feedback_rows": unused,
            "threshold": threshold,
        }
    proc = subprocess.Popen(
        [sys.executable, "scripts/retrain.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {
        "status": "started",
        "pid": proc.pid,
        "unused_feedback_rows": unused,
    }


@router.post("/explain")
def explain(task: JiraTask):
    if not task.summary.strip():
        raise HTTPException(status_code=400, detail="Summary cannot be empty")
    try:
        feature_df = _extract_features(
            task.summary, task.description, task.region, task.subsystem, task.commitments
        )
        pool = Pool(data=feature_df, cat_features=CAT_FEATURES)

        model = ml_models["est_folds"][0]
        shap_vals = model.get_feature_importance(pool, type="ShapValues")
        base_value_log = float(shap_vals[0, -1])
        feature_contribs = shap_vals[0, :-1]

        preds_log = [m.predict(pool) for m in ml_models["est_folds"]]
        prediction_days = float(np.expm1(np.mean(preds_log)))

        feature_names = list(feature_df.columns)
        pairs = sorted(zip(feature_names, feature_contribs), key=lambda x: abs(x[1]), reverse=True)

        return {
            "base_value": round(float(np.expm1(base_value_log)), 2),
            "prediction": round(prediction_days, 1),
            "top_features": [
                {"feature": name, "shap_value": round(float(val), 4)}
                for name, val in pairs[:10]
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
