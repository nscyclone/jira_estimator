import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="JiraEstimator", layout="wide")
st.title("JiraEstimator — Advisory Effort Prediction")
st.caption("Data-driven second opinion for sprint planning.")

tab_predict, tab_metrics, tab_feedback = st.tabs(["Predict", "Metrics", "Feedback"])

# ── TAB 1: PREDICT ────────────────────────────────────────────────────────────
with tab_predict:
    st.subheader("Predict effort for a Jira ticket")

    summary = st.text_input("Summary *", "Реализовать выгрузку СЭМД протокола осмотра врача")
    description = st.text_area("Description", "Добавить формирование CDA R2 по утверждённому шаблону. Шаблон согласован с главным врачом.", height=100)
    col1, col2, col3 = st.columns(3)
    with col1:
        region = st.text_input("Region", "БАЗОВЫЙ")
    with col2:
        subsystem = st.text_input("Subsystem", "СЭМД/Выгрузка")
    with col3:
        commitments = st.text_input("Commitments", "ТУР")

    if st.button("Predict", type="primary"):
        payload = {
            "summary": summary, "description": description,
            "region": region, "subsystem": subsystem, "commitments": commitments,
        }
        try:
            resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            c1, c2 = st.columns(2)
            c1.metric("Base Estimate", f"{data['predicted_time_hours']:.1f} h",
                      f"{data['predicted_time_hours'] / 8:.1f} FTE-days")
            c2.metric("Risk-Adjusted", f"{data['adjusted_time_with_buffer_hours']:.1f} h")

            st.subheader("Risk Profile")
            risk = data["risk_profile"]
            col_l, col_m, col_c = st.columns(3)
            col_l.metric("Low risk", f"{risk['low_risk_prob_pct']:.0f}%")
            col_m.metric("Medium risk", f"{risk['medium_risk_prob_pct']:.0f}%")
            col_c.metric("Critical risk", f"{risk['critical_risk_prob_pct']:.0f}%",
                         delta=None if risk['critical_risk_prob_pct'] < 20 else "⚠ High",
                         delta_color="inverse")

            # SHAP
            explain_resp = requests.post(f"{API_BASE}/explain", json=payload, timeout=15)
            if explain_resp.status_code == 200:
                ex = explain_resp.json()
                st.subheader(f"Why this estimate? (base: {ex['base_value']:.2f} days)")
                import pandas as pd
                shap_df = pd.DataFrame(ex["top_features"]).set_index("feature")
                st.bar_chart(shap_df["shap_value"])
                st.caption("Positive = increases estimate, Negative = reduces estimate (log-space contributions)")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Start the server: `uvicorn app:app --port 8000`")
        except Exception as e:
            st.error(f"Error: {e}")

# ── TAB 2: METRICS ────────────────────────────────────────────────────────────
with tab_metrics:
    st.subheader("Model & Business Metrics")

    if st.button("Refresh metrics"):
        try:
            resp = requests.get(f"{API_BASE}/metrics", timeout=5)
            resp.raise_for_status()
            m = resp.json()

            st.markdown(
                f"**Model revision:** `{m.get('model_revision') or 'N/A'}` &nbsp;|&nbsp; "
                f"**Published:** {m.get('model_published_at') or 'N/A'} &nbsp;|&nbsp; "
                f"**Trigger:** {m.get('model_trigger') or 'N/A'}"
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R² (test)", f"{m['model_r2']:.3f}" if m.get("model_r2") else "N/A",
                      help="Proportion of effort variance explained. ~0.14 is the signal ceiling for ticket text.")
            c2.metric("MAE (FTE-days)", f"{m['model_mae']:.2f}" if m.get("model_mae") else "N/A")
            c3.metric("Overrun Recall", f"{m['model_overrun_recall']:.0%}" if m.get("model_overrun_recall") else "N/A",
                      help="% of critical overruns detected early. ~59%.")
            c4.metric("Hours Saved / Sprint", f"{m['estimated_hours_saved_per_sprint']:.1f} h",
                      help="Based on feedbacks collected × 15 min saved vs. planning poker")

            st.divider()
            st.caption(f"Feedbacks collected: {m['feedbacks_collected']} &nbsp;|&nbsp; "
                       f"Feedback accuracy (±25%): {m.get('feedback_accuracy_within_25pct') or 'N/A'} &nbsp;|&nbsp; "
                       f"Mean delta: {m.get('feedback_mean_delta_pct') or 'N/A'}%")

            if m.get("model_overrun_recall"):
                st.info(
                    f"**Honest disclosure:** Risk classifier catches {m['model_overrun_recall']:.0%} of critical "
                    f"overruns but has ~70% false positive rate. Use as early-warning signal, not a decision gate."
                )
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Start the server: `uvicorn app:app --port 8000`")
        except Exception as e:
            st.error(f"Error: {e}")

# ── TAB 3: FEEDBACK ───────────────────────────────────────────────────────────
with tab_feedback:
    st.subheader("Submit actual hours after task completion")
    st.caption("This data is used to retrain the model when 50 feedback entries are collected.")

    fb_summary = st.text_input("Ticket Summary *", key="fb_summary")
    fb_description = st.text_area("Description", "", height=80, key="fb_description")
    col1, col2 = st.columns(2)
    with col1:
        fb_predicted = st.number_input("Predicted (days)", min_value=0.0, value=1.5, step=0.5)
    with col2:
        fb_actual = st.number_input("Actual (days)", min_value=0.0, value=2.0, step=0.5)
    col3, col4, col5 = st.columns(3)
    with col3:
        fb_region = st.text_input("Region", "БАЗОВЫЙ", key="fb_region")
    with col4:
        fb_subsystem = st.text_input("Subsystem", "", key="fb_subsystem")
    with col5:
        fb_commitments = st.text_input("Commitments", "SLA", key="fb_commitments")

    if st.button("Submit Feedback", type="primary"):
        if not fb_summary.strip():
            st.warning("Summary is required.")
        else:
            try:
                resp = requests.post(f"{API_BASE}/feedback", json={
                    "summary": fb_summary,
                    "description": fb_description,
                    "region": fb_region,
                    "subsystem": fb_subsystem,
                    "commitments": fb_commitments,
                    "predicted_days": fb_predicted,
                    "actual_days": fb_actual,
                }, timeout=5)
                resp.raise_for_status()
                result = resp.json()
                unused = result.get("unused_feedback_count", result["feedback_count"])
                need_more = max(0, 50 - unused)
                retrain_note = "🔄 Ready to retrain!" if result["retrain_ready"] else f"Need {need_more} more to trigger retraining"
                delta_str = f"{result['delta_pct']:.1f}%" if result["delta_pct"] is not None else "N/A"
                st.success(
                    f"Feedback saved! Delta: {delta_str} | "
                    f"Total collected: {result['feedback_count']} | "
                    f"{retrain_note}"
                )
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API.")
            except Exception as e:
                st.error(f"Error: {e}")
