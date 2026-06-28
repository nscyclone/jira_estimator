# Jira Effort Estimator

**Data-driven advisory tool for software task estimation.**  
Predicts expected effort in workdays from a Jira ticket's text and metadata, giving teams an objective anchor before planning poker and assignee selection.

![CI](https://github.com/nscyclone/jira_estimator/actions/workflows/ci.yml/badge.svg)

---

## Problem

Software teams systematically underestimate task effort. Planning Poker relies entirely on human judgment — subject to anchoring bias, social pressure, and the optimism typical of estimation sessions. Commercial tools (COCOMO II, Jira's built-in estimate) either require manual calibration or ignore the team's own historical data.

This project asks: **can a team's past Jira backlog predict future effort well enough to serve as a useful second opinion?**

The honest answer from our research: **partially** — and the constraints are worth understanding.

---

## Key Findings

| Finding | Detail |
|---|---|
| Text-only R² ceiling | ~0.17 — consistent with Jørgensen SLR and Nassif (2019) |
| BERT vs. LSA | ΔR² = 0.003, cost ratio = 1:30 — BERT adds no signal |
| Root cause | ~85% of effort variance is human factor, tech debt, team dynamics — invisible in ticket text |
| Production decision | Deploy as **advisory anchor**, not autopilot; ensemble uncertainty quantifies the risk buffer |

> The absence of a strong model is itself a scientific result. Knowing the Bayes Error Rate of your signal prevents wasted engineering effort on architecture complexity that cannot close an irreducible gap.

---

## Dataset

Production Jira backlog from a Russian medical information system (MIS), exported via REST API. All ticket text is in Russian.

| Property | Value |
|---|---|
| Raw export | 100 000 tickets |
| After filtering (story points + time logged required) | 87 443 (12.6% dropped) |
| Language | Russian |
| Train / Val / Test split | 69 954 / 8 745 / 8 744 (80/10/10, stratified) |

### Target: `logged_days`

Actual developer time logged, converted from seconds to 8-hour workdays.

| Statistic | Value |
|---|---|
| Mean | 1.94 FTE-days |
| Median | 1.06 FTE-days |
| Std | 2.74 |
| p95 / p99 / Max | 6.6 / 12.3 / 106 FTE-days |

The distribution is heavily right-skewed (median ≪ mean). Model trains on `log1p(logged_days)`; predictions are converted back via `expm1`.

### Risk label distribution

Derived from `logged_days / story_points` ratio.

| Class | Condition | Count | Share |
|---|---|---|---|
| 0 — Low | ratio ≤ 1.0 | 50 117 | 57.3% |
| 1 — Medium | 1.0 < ratio ≤ 1.5 | 19 400 | 22.2% |
| 2 — Critical | ratio > 1.5 | 17 926 | 20.5% |

### Categorical features

| Feature | Unique values | Notes |
|---|---|---|
| `region` | 917 | БАЗОВЫЙ covers 77% of tickets (default region in MIS deployments) |
| `subsystem` | 353 | Reflects MIS module hierarchy: e.g. Отчеты/Отчеты, Поликлиника |
| `commitments` | 15 | SLA (32%), ТЗР (10%), Рефакторинг (6%) are top-3 |

---

## Architecture

```mermaid
flowchart TD
    subgraph train["Training Pipeline  (scripts/)"]
        A([Jira API]) -->|get_jira_issues| B[(seed.csv)]
        B -->|prepare_data| C[(dataset.csv)]
        C -->|split_data| D[(train / val / test)]
        D -->|extract_bm25_lsa\nTF-IDF · SVD 32D| E[(embeddings)]
        E -->|train_catboost_estimate| F[(estimate × 5 folds)]
        E -->|train_catboost_risk| G[(risk × 5 folds)]
        F & G --> H[(MLflow · mlruns/)]
    end

    subgraph service["Inference Service"]
        I([Client]) -->|HTTP| J[FastAPI]
        J --> K[LSA transform\n+ text features]
        K --> L[CatBoost Ensemble]
        L -->|effort · risk · SHAP| J
        J -->|POST /feedback| M[(SQLite\nfeedback.db)]
    end

    subgraph loop["Feedback Loop"]
        M -->|≥ 50 rows| N[retrain.py]
        N -->|updated models| F
    end

    F --> L
    G --> L
    E -.->|pipeline.pkl| K
```

### Feature Set (42 total)

| Group | Features | Count |
|---|---|---|
| LSA text embedding | TF-IDF → SVD(32D) over `summary + description` | 32 |
| Structural flags | `has_description`, `has_code_block`, `is_dev_task`, `is_test_task`, `is_analysis_task` | 5 |
| Length metrics | `text_len`, `word_count` | 2 |
| Jira metadata | `region`, `subsystem`, `commitments` (CatBoost native categorical) | 3 |

### Models

**Effort regressor** — `CatBoostRegressor`  
`iterations=15000, depth=8, lr=0.03, l2_leaf_reg=6.0, loss=RMSE`  
Target: `log1p(logged_days)` — log transform stabilises the 103 FTE outlier.  
Per-fold outlier removal: 1st–99th percentile of training labels only.

**Risk classifier** — `CatBoostClassifier`  
`iterations=4000, depth=6, lr=0.03, loss=MultiClass, auto_class_weights=Balanced`  
3 classes derived from `logged_days / story_points` ratio:

| Class | Condition | Meaning |
|---|---|---|
| 0 — Low | ratio ≤ 1.0 | Delivered on or under estimate |
| 1 — Medium | 1.0 < ratio ≤ 1.5 | Mild overrun |
| 2 — Critical | ratio > 1.5 | Significant overrun |

### Inference: Risk Buffer

```
adjusted_hours = base_hours × (1 + 0.15 × P_medium + 0.50 × P_critical)
```

---

## Results

### Regression (effort estimation)

| Metric | Value |
|---|---|
| CV R² (5-Fold, mean ± std) | 0.159 ± 0.016 |
| Test R² | 0.184 |
| Test MAE | 1.19 FTE |
| Dataset mean | 1.97 FTE |
| MdAPE | 32.6% |

### Baseline Comparison

| Baseline | Test R² | Test MAE |
|---|---|---|
| Global median | −0.045 | 1.399 FTE |
| Median by subsystem | 0.010 | 1.384 FTE |
| Median by region | −0.032 | 1.394 FTE |
| **CatBoost 5-fold ensemble** | **0.184** | **1.193 FTE** |

Naive baselines (predict the per-group median) explain zero or negative variance on this dataset. The model's R²=0.184 represents genuine signal from ticket text and metadata — 17× better than the strongest naive baseline.

Reproduced with `python scripts/compute_baseline.py`.

### Architecture Comparison

| Architecture | Test R² | MAE | RAM | Train time/fold | Monthly infra cost |
|---|---|---|---|---|---|
| RuBERT-Heavy (768D + 7) | 0.147 | 1.24 FTE | ~6 GB | ~40 min | ~$150 |
| **LSA-Light (32D + 7 + 3 cat)** | **0.184** | **1.19 FTE** | **~200 MB** | **~2 min** | **~$5** |

ΔR² = 0.037 in favour of LSA-Light at 1/30 the cost. LSA-Light is the production choice.

### Risk Classifier

| Metric | Value |
|---|---|
| Recall (Critical class) | 59% |
| Precision (Risk classes) | 27–31% |

**Interpretation**: The classifier catches ~6 in 10 critical overruns before they happen — useful as an early-warning signal, not as a decision gate. ~70% of alerts are false positives; use accordingly.

---

## Quickstart (local)

### Prerequisites

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires trained model files in `models/` and `embeddings/bm25_lsa_pipeline.pkl`.  
If starting fresh, run the [full pipeline](#full-pipeline-local) first.

### Seed baseline metrics and start the API

```bash
# Seed model_runs table with baseline metrics (idempotent)
python scripts/seed_model_runs.py

# Start inference server
uvicorn app:app --host 0.0.0.0 --port 8000

# (Optional) Start Streamlit demo
streamlit run demo.py
```

### Docker

```bash
docker build -t jira-estimator .
docker run -p 8000:8000 jira-estimator
```

---

## MLOps

### Experiment Tracking

All training runs are logged to MLflow (`file:./mlruns`, no server required):

```bash
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

Each run records hyperparameters, per-fold CV scores, test metrics, and a SHAP summary plot.

### Feedback Loop

```
User submits feedback via /feedback or Streamlit
    │
    ▼
feedback.db (is_used_for_training = 0)
    │
    ▼ (when unused count ≥ 50)
retrain.py
    │
    ├── Augments CV training set with feedback rows
    ├── Retrains estimate model (5-Fold CatBoostRegressor)
    ├── Logs new MLflow run (trigger='feedback')
    └── Updates model_runs → /health and /metrics reflect new revision
```

```bash
# Check if retraining threshold is reached (dry run)
python scripts/retrain.py --dry-run

# Trigger retraining
python scripts/retrain.py
```

### Seeding Baseline Metrics

For local development or demo setup without a full training run:

```bash
python scripts/seed_model_runs.py          # idempotent, skips if already seeded
python scripts/seed_model_runs.py --force  # re-seed even if rows exist
python scripts/seed_model_runs.py --db path/to/custom.db
```

---

## API Reference

### `GET /health`

```json
{
  "status": "ok",
  "folds_loaded": true,
  "model_revision": "48d6f30",
  "model_published_at": "2026-06-27T20:53:27+00:00"
}
```

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "summary": "Реализовать выгрузку СЭМД протокола осмотра врача",
    "description": "Добавить формирование CDA R2 по утверждённому шаблону.",
    "region": "БАЗОВЫЙ",
    "subsystem": "СЭМД/Выгрузка",
    "commitments": "ТЗР"
  }'
```

```json
{
  "predicted_time_hours": 12.4,
  "adjusted_time_with_buffer_hours": 15.1,
  "prediction_std_days": 0.08,
  "risk_profile": {
    "low_risk_prob_pct": 61.3,
    "medium_risk_prob_pct": 24.7,
    "critical_risk_prob_pct": 14.0
  }
}
```

`prediction_std_days` — standard deviation of the 5-fold ensemble predictions in day-space. Use it as a free uncertainty estimate: high std means folds disagree on this ticket.

### `POST /explain`

Returns SHAP feature contributions for the effort estimate (fold 0, log-space values).

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "summary": "Реализовать выгрузку СЭМД протокола осмотра врача",
    "description": "Добавить формирование CDA R2 по утверждённому шаблону.",
    "region": "БАЗОВЫЙ",
    "subsystem": "СЭМД/Выгрузка",
    "commitments": "ТЗР"
  }'
```

```json
{
  "base_value": 1.83,
  "prediction": 1.6,
  "top_features": [
    {"feature": "region", "shap_value": 0.1842},
    {"feature": "emb_7",  "shap_value": -0.0913}
  ]
}
```

`shap_value` is in log-space: positive = increases estimate, negative = decreases. `exp(shap_value)` ≈ multiplicative effect on days.

### `POST /feedback`

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "summary": "Реализовать выгрузку СЭМД протокола осмотра врача",
    "description": "Добавить формирование CDA R2 по утверждённому шаблону.",
    "region": "БАЗОВЫЙ",
    "subsystem": "СЭМД/Выгрузка",
    "commitments": "SLA",
    "predicted_days": 1.5,
    "actual_days": 2.0
  }'
```

```json
{
  "delta_pct": 25.0,
  "feedback_count": 12,
  "unused_feedback_count": 12,
  "retrain_ready": false
}
```

### `GET /metrics`

Dynamic metrics from SQLite — updates automatically after each training run.

```json
{
  "model_revision": "48d6f30",
  "model_published_at": "2026-06-27T20:53:27+00:00",
  "model_trigger": "manual",
  "model_r2": 0.184,
  "model_mae": 1.193,
  "model_overrun_recall": 0.59,
  "feedbacks_collected": 12,
  "feedback_mean_delta_pct": 18.3,
  "feedback_accuracy_within_25pct": 0.667,
  "estimated_hours_saved_per_sprint": 3.0
}
```

### `POST /retrain`

Checks whether unused feedback rows have reached the retraining threshold (default 50). If so, launches `scripts/retrain.py` as a background process and returns immediately.

```bash
curl -X POST http://localhost:8000/retrain
```

```json
{ "status": "skipped", "unused_feedback_rows": 12, "threshold": 50 }
```

```json
{ "status": "started", "pid": 84312, "unused_feedback_rows": 51 }
```

The process runs in the background — monitor progress via `GET /metrics` (model revision updates after completion) or MLflow UI.

---

## Full Pipeline (local)

```bash
# 1. Export raw backlog from Jira
python scripts/get_jira_issues.py           # → data/seed.csv

# 2. Feature engineering + risk labelling
python scripts/prepare_data.py              # → data/dataset.csv

# 3. Stratified train/val/test split
python scripts/split_data.py                # → data/train.csv, val.csv, test.csv

# 4. Fit LSA embedding pipeline
python scripts/extract_bm25_lsa.py          # → embeddings/*.npy + bm25_lsa_pipeline.pkl

# 5. Train regression ensemble (logs to MLflow)
python scripts/train_catboost_estimate.py   # → models/estimate_fold_{0..4}.cbm

# 6. Train risk classification ensemble (logs to MLflow)
python scripts/train_catboost_risk.py       # → models/risk_fold_{0..4}.cbm

# 7. Start inference server
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Project Structure

```
├── app.py                      # FastAPI server (predict, explain, feedback, metrics, health)
├── config.py                   # Centralised config (paths, hyperparams, MLflow, SQLite)
├── feature_engineering.py      # Shared keyword lists + compute_text_features()
├── feedback.py                 # SQLite store — feedback and model_runs tables
├── load_catboost_data.py       # Shared data loader for CatBoost trainers
├── demo.py                     # Streamlit demo (Predict / Metrics / Feedback tabs)
│
├── scripts/
│   ├── get_jira_issues.py          # Jira REST API export (requires JIRA_URL/JIRA_PROJECT/JIRA_COOKIE env vars)
│   ├── prepare_data.py             # Raw → clean dataset with features + risk labels
│   ├── split_data.py               # Stratified train/val/test split
│   ├── extract_bm25_lsa.py         # TF-IDF + TruncatedSVD pipeline
│   ├── train_catboost_estimate.py  # 5-Fold CatBoostRegressor training + MLflow + SHAP
│   ├── train_catboost_risk.py      # 5-Fold CatBoostClassifier training + MLflow
│   ├── retrain.py                  # Feedback-triggered retraining (threshold: 50 rows)
│   ├── seed_model_runs.py          # Seed model_runs with baseline metrics for local dev
│   ├── seed_synthetic_feedback.py  # Seed ~20 synthetic feedback rows for demo
│   ├── compute_baseline.py         # Naive baseline comparison (median, by-subsystem)
│   └── analyze_errors.py           # Per-subsystem/region MAE + worst predictions
│
├── notebooks/
│   └── pipeline.ipynb              # End-to-end walkthrough: seed → train → infer → retrain
│
├── archive/                        # BERT experiments (not used in production)
│   ├── dataset.py                  # JiraDataset (PyTorch)
│   ├── estimate_model.py           # RuBERT regression head
│   ├── risk_model.py               # RuBERT classification head
│   ├── estimate_train.py           # RuBERT training loop
│   ├── extract_embeddings.py       # RuBERT embedding extraction
│   ├── load_estimate_data.py       # BERT data loader (estimate)
│   └── load_risk_data.py           # BERT data loader (risk)
│
├── tests/
│   └── test_feedback.py        # SQLite store unit tests (no ML deps)
├── .github/workflows/ci.yml    # CI: ruff lint + pytest on Python 3.12
├── ruff.toml                   # Linter config
│
├── embeddings/                 # bm25_lsa_pipeline.pkl + *.npy arrays (not committed)
├── models/                     # Trained .cbm fold models (not committed)
└── data/                       # CSV schemas committed; full data kept locally (NDA)
```

---

## Competitive Context

| Tool | Approach | Limitation |
|---|---|---|
| Planning Poker | Expert judgment | Anchoring bias, social pressure, no historical grounding |
| COCOMO II | Function point formula | Requires manual calibration; ignores team-specific patterns |
| Jira built-in estimate | Manual field | No predictive component; teams fill it after discussion anyway |
| **This tool** | Team's own backlog history | R² = 0.14–0.17; honest advisory anchor, not an oracle |

---

## AI Tools Used

- **Claude Code** (Anthropic) — code review, refactoring, MLOps architecture (MLflow integration, feedback loop design, SHAP explainability, Streamlit demo, GitHub Actions CI).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Inference API | FastAPI + Uvicorn |
| ML models | CatBoost 1.2.5 |
| Text pipeline | scikit-learn TF-IDF + TruncatedSVD |
| Experiment tracking | MLflow 2.13.0 (file-based) |
| Feedback store | SQLite (stdlib) |
| Explainability | CatBoost native SHAP |
| Demo UI | Streamlit |
| Data | pandas + numpy |
| Containerisation | Docker |
| CI | GitHub Actions (ruff + pytest) |
| Experiment infra | BERT experiments: PyTorch 2.3 + HuggingFace Transformers |
