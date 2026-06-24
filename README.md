# Jira Effort Estimator

**Data-driven advisory tool for software task estimation.**  
Predicts expected effort in workdays from a Jira ticket's text and metadata, giving teams an objective anchor before planning poker and assignee selection.

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

## Architecture

```
Jira Backlog
    │
    ▼
get_jira_issues.py       ← REST API export (JQL, pagination)
    │
    ▼
prepare_data.py          ← feature engineering + risk label creation
    │
    ▼
split_data.py            ← stratified 80/10/10 train/val/test split
    │
    ▼
extract_bm25_lsa.py      ← TF-IDF (sublinear_tf, min_df=2, 25k vocab)
                            + TruncatedSVD (32 components) → LSA embeddings
    │
    ├── train_catboost_estimate.py   ← 5-Fold KFold regression ensemble
    └── train_catboost_risk.py       ← 5-Fold KFold classification ensemble
                │
                ▼
            app.py (FastAPI)         ← inference: ensemble mean + risk buffer
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

The buffer converts probabilistic risk into a conservative planning estimate — not a hard deadline, but an upper bound the team should discuss before sprint commitment.

---

## Results

### Regression (effort estimation)

| Metric | Value |
|---|---|
| CV R² (5-Fold, mean ± std) | 0.166 ± 0.007 |
| Test R² | 0.144 |
| Test MAE | 1.23 FTE |
| Test RMSE | — |
| Dataset mean | 1.97 FTE |
| Dataset std | 3.45 FTE |
| MdAPE | 32.6% |

### Architecture Comparison

| Architecture | Test R² | MAE | RAM | Train time/fold | Monthly infra cost |
|---|---|---|---|---|---|
| RuBERT-Heavy (768D + 7) | 0.147 | 1.24 FTE | ~6 GB | ~40 min | ~$150 |
| **LSA-Light (32D + 7 + 3 cat)** | **0.144** | **1.23 FTE** | **~200 MB** | **~2 min** | **~$5** |

ΔR² = 0.003 in favour of BERT at 30× the cost. LSA-Light is the production choice.

### Risk Classifier

| Metric | Value |
|---|---|
| Recall (Critical class) | 59% |
| Precision (Risk classes) | 27–31% |

**Interpretation**: The classifier catches ~6 in 10 critical overruns before they happen — useful as an early-warning signal, not as a decision gate.

### Critical Limitations

- MAE = 1.23 FTE on a mean of 1.97 FTE (~62% relative error). **Do not use as an automated sprint planner.**
- 70% of risk alerts are false positives. Extended use without UX care will cause alert fatigue.
- Model generalises only to backlogs with similar domain, team, and workflow. Retraining on your own data is required.

---

## Quickstart (Docker)

Requires trained model files in `models/` and `embeddings/bm25_lsa_pipeline.pkl`.

```bash
docker build -t jira-estimator .
docker run -p 8000:8000 jira-estimator
```

Health check:
```bash
curl http://localhost:8000/health
# {"status":"ok","folds_loaded":true}
```

Predict:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "summary": "Implement OAuth2 login for the mobile client",
    "description": "Need to add OAuth2 authorization flow. Backend already has the endpoint.",
    "region": "MOSCOW",
    "subsystem": "Auth/Mobile",
    "commitments": "Q3"
  }'
```

Response:
```json
{
  "predicted_time_hours": 12.4,
  "adjusted_time_with_buffer_hours": 15.1,
  "risk_profile": {
    "low_risk_prob_pct": 61.3,
    "medium_risk_prob_pct": 24.7,
    "critical_risk_prob_pct": 14.0
  }
}
```

---

## Full Pipeline (local)

### Prerequisites

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set your Jira session cookie:
```bash
export JIRA_COOKIE="JSESSIONID=your_session_id_here"
export JIRA_URL="https://your-jira-instance.example.com"   # optional, has default
```

### Step-by-step

```bash
# 1. Export raw backlog from Jira
python get_jira_issues.py           # → data/seed.csv

# 2. Feature engineering + risk labelling
python prepare_data.py              # → data/dataset.csv

# 3. Stratified train/val/test split
python split_data.py                # → data/train.csv, val.csv, test.csv

# 4. Fit LSA embedding pipeline
python extract_bm25_lsa.py          # → embeddings/*.npy + bm25_lsa_pipeline.pkl

# 5. Train regression ensemble
python train_catboost_estimate.py   # → models/estimate_fold_{0..4}.cbm

# 6. Train risk classification ensemble
python train_catboost_risk.py       # → models/risk_fold_{0..4}.cbm

# 7. Start inference server
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Project Structure

```
├── app.py                      # FastAPI inference server
├── config.py                   # Centralised config (paths, hyperparams)
├── feature_engineering.py      # Shared keyword lists + compute_text_features()
├── dataset.py                  # Unified JiraDataset (PyTorch, for BERT experiments)
├── prepare_data.py             # Raw → clean dataset with features + risk labels
├── split_data.py               # Stratified train/val/test split
├── extract_bm25_lsa.py         # TF-IDF + TruncatedSVD pipeline
├── extract_embeddings.py       # RuBERT CLS-token embedding extraction (archived)
├── load_catboost_data.py       # Shared data loader for both CatBoost trainers
├── train_catboost_estimate.py  # 5-Fold CatBoostRegressor training + evaluation
├── train_catboost_risk.py      # 5-Fold CatBoostClassifier training + evaluation
├── estimate_model.py           # RuBERT regression head (archived)
├── risk_model.py               # RuBERT classification head (archived)
├── estimate_train.py           # RuBERT training loop (archived)
├── get_jira_issues.py          # Jira REST API export script
├── Dockerfile                  # Production inference image
├── requirements.txt
├── embeddings/                 # LSA pipeline + embedding arrays
├── models/                     # Trained .cbm fold models
└── data/                       # CSV datasets (not committed)
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

- **Claude Code** (Anthropic) — automated code review identifying a data leakage bug (`global_val_pool` passed as `eval_set` across all folds), `num_risk_classes` config mismatch, duplicate data loaders, and insecure credential handling; applied refactoring to consolidate duplicated feature engineering logic into `feature_engineering.py`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Inference API | FastAPI + Uvicorn |
| ML models | CatBoost 1.2.5 |
| Text pipeline | scikit-learn TF-IDF + TruncatedSVD |
| Data | pandas + numpy |
| Containerisation | Docker (python:3.10-slim) |
| Experiment infra | BERT experiments: PyTorch 2.3 + HuggingFace Transformers |
