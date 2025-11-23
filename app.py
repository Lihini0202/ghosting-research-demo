# app.py - Modern HOLO themed Streamlit app for Ghosting Research
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import gdown
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(page_title="HOLO — Ghosting Research", layout="wide", initial_sidebar_state="expanded")

# --- THEME COLORS ---
HOLO_PURPLE = "#5a0891"
ACCENT = HOLO_PURPLE
CARD_BG = "#ffffff"
BG = "#f8f8fb"

# --- CUSTOM CSS (modern, clean) ---
st.markdown(
    f"""
    <style>
    :root {{
      --accent: {ACCENT};
      --bg: {BG};
      --card: {CARD_BG};
      --muted: #6b7280;
    }}
    .reportview-container .main {{
      background-color: var(--bg);
    }}
    header .decoration {{
      display: none;
    }}
    .stApp > header {{ visibility: hidden; }}
    .holo-title {{
      display:flex;
      align-items:center;
      gap:12px;
      font-family: 'Poppins', sans-serif;
    }}
    .card {{
      background: var(--card);
      border-radius: 12px;
      padding: 18px;
      box-shadow: 0 6px 18px rgba(18, 18, 18, 0.06);
      border: 1px solid rgba(92, 88, 126, 0.06);
    }}
    .metric {{
      font-size: 22px;
      color: var(--accent);
      font-weight: 600;
    }}
    .small-muted {{
      color: var(--muted);
      font-size: 13px;
    }}
    /* Tabs styling fix */
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
      height:46px;
      background:#f0f2f6;
      color:#111827;
      border-radius:10px 10px 0 0;
      padding-top:8px;
      padding-bottom:8px;
    }}
    .stTabs [aria-selected="true"] {{
      background: #ffffff;
      border-top: 3px solid var(--accent);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- HEADER ---
left_col, right_col = st.columns([3, 1])
with left_col:
    st.markdown(f"<div class='holo-title'><h1 style='margin:0'>👻 HOLO — Ghosting Prediction Research</h1></div>", unsafe_allow_html=True)
    st.markdown("**Candidate:** [Your Name]  •  **Topic:** Behavioral Analysis of Ghosting — modern visualization & simulator")
with right_col:
    st.markdown("<div class='small-muted'>Built with ❤️ • HOLO Theme</div>", unsafe_allow_html=True)

st.markdown("---")

# --- HELPER: load assets safely ---
@st.cache_resource
def load_assets():
    """
    Loads performance JSON and model files if available.
    If missing, it falls back to a built-in summary `all_models_data`.
    Replace file IDs and filenames as needed.
    """
    # try load perf JSON
    perf = None
    model = None
    model_columns = None
    scaler = None

    # Attempt to load local JSON
    perf_path = Path("model_performance.json")
    if perf_path.exists():
        try:
            with open(perf_path, "r") as f:
                perf = json.load(f)
        except Exception:
            perf = None

    #  download model from Google Drive if env variable or id provided
    
    DRIVE_FILE_ID = os.environ.get("1gAogfnZDcpuSOTLa0UTD4tvXM0Vk0Jp2", "")  
    MODEL_FILENAME = "ghosting_risk_model.pkl"

    if DRIVE_FILE_ID and not Path(MODEL_FILENAME).exists():
        try:
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            gdown.download(url, MODEL_FILENAME, quiet=True)
        except Exception:
            pass

    # load model if present
    if Path(MODEL_FILENAME).exists():
        try:
            model = joblib.load(MODEL_FILENAME)
        except Exception:
            model = None

    # load columns and scaler if present
    if Path("model_columns.pkl").exists():
        try:
            model_columns = joblib.load("model_columns.pkl")
        except Exception:
            model_columns = None

    if Path("scaler.pkl").exists():
        try:
            scaler = joblib.load("scaler.pkl")
        except Exception:
            scaler = None

    return perf, model, model_columns, scaler

perf_data, model, model_columns, scaler = load_assets()

# --- FALLBACK: if perf_data missing use hard-coded summary (keeps app functional) ---
all_models_data = {
    "Random Forest": {"Accuracy": 0.9352, "Precision": 0.89, "Recall": 0.87, "F1": 0.88, "AUC": 0.9352},
    "Ensemble": {"Accuracy": 0.9337, "Precision": 0.89, "Recall": 0.86, "F1": 0.87, "AUC": 0.9337},
    "Gradient Boosting": {"Accuracy": 0.9319, "Precision": 0.89, "Recall": 0.86, "F1": 0.87, "AUC": 0.9319},
    "Logistic Regression": {"Accuracy": 0.8953, "Precision": 0.92, "Recall": 0.84, "F1": 0.88, "AUC": 0.8953},
    "Naive Bayes": {"Accuracy": 0.7692, "Precision": 0.75, "Recall": 0.70, "F1": 0.72, "AUC": 0.7692},
    "SVM": {"Accuracy": 0.5000, "Precision": 0.50, "Recall": 0.50, "F1": 0.50, "AUC": 0.5000},
    "K-NN": {"Accuracy": 0.4871, "Precision": 0.49, "Recall": 0.48, "F1": 0.48, "AUC": 0.4871},
}

if perf_data and isinstance(perf_data, dict):
    # prefer perf_data if it contains model-level metrics
    # Attempt to extract model summary keys if available
    extracted = {}
    for k, v in perf_data.get("models", {}).items():
        # expect v to contain metrics; adapt if structure differs
        extracted[k] = {
            "Accuracy": v.get("accuracy", np.nan),
            "Precision": v.get("precision", np.nan),
            "Recall": v.get("recall", np.nan),
            "F1": v.get("f1", np.nan),
            "AUC": v.get("auc", np.nan),
        }
    if extracted:
        all_models_data.update(extracted)

# --- UTILS: plot functions ---
def plot_bar_metrics(df_metrics, metric="AUC"):
    fig = px.bar(
        df_metrics.sort_values(by=metric, ascending=True),
        x=metric, y="Model", orientation="h",
        text=metric,
        title=f"{metric} Comparison",
        color=metric,
        color_continuous_scale="Purples",
        range_x=[0, 1]
    )
    fig.update_layout(coloraxis_showscale=False, height=420, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def plot_scatter_metrics(df_metrics, x_metric="Accuracy", y_metric="Recall"):
    fig = px.scatter(
        df_metrics, x=x_metric, y=y_metric, text="Model",
        size="F1", title=f"{x_metric} vs {y_metric} (bubble size = F1)",
        labels={x_metric: x_metric, y_metric: y_metric}
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def plot_hist_metric(df_metrics, metric="AUC"):
    # create small distribution from the column values for histogram demonstration
    values = df_metrics[metric].dropna().values
    # If single value, create a small jittered distribution so histogram appears
    if len(values) == 1:
        values = np.repeat(values, 20) + np.random.normal(0, 0.005, 20)
    fig = px.histogram(values, nbins=10, title=f"{metric} Distribution (approx.)")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
    return fig

def metric_card_html(label, value, delta=None):
    delta_html = f"<div style='color:green; font-size:12px'>{delta}</div>" if delta else ""
    return f"""
        <div class="card" style="text-align:center">
            <div style="font-size:13px; color:#6b7280">{label}</div>
            <div class="metric" style="margin-top:6px">{value}</div>
            {delta_html}
        </div>
    """

# --- Prepare DataFrame from summary dict ---
df_compare = pd.DataFrame.from_dict(all_models_data, orient="index").reset_index().rename(columns={"index": "Model"})
# Ensure numeric types
for col in ["Accuracy", "Precision", "Recall", "F1", "AUC"]:
    df_compare[col] = pd.to_numeric(df_compare[col], errors="coerce")

# --- NAVIGATION TABS ---
tab_names = ["🏆 Leaderboard", "🌲 Random Forest", "🚀 Gradient Boosting", "📈 Logistics", "🔮 Ensemble", "🧪 Simulator"]
tabs = st.tabs(tab_names)

# ------------------------------
# TAB 1: Leaderboard
# ------------------------------
with tabs[0]:
    st.header("Model Leaderboard")
    left, right = st.columns([3, 1.2])
    with left:
        st.plotly_chart(plot_bar_metrics(df_compare, metric="AUC"), use_container_width=True)
        st.plotly_chart(plot_scatter_metrics(df_compare, x_metric="Accuracy", y_metric="Recall"), use_container_width=True)
    with right:
        st.markdown(metric_card_html("Top Model", df_compare.sort_values("AUC", ascending=False).iloc[0]["Model"]))
        top_metrics = df_compare.sort_values("AUC", ascending=False).iloc[0]
        st.markdown(metric_card_html("AUC", f"{top_metrics['AUC']:.4f}"))
        st.markdown(metric_card_html("Accuracy", f"{top_metrics['Accuracy']:.1%}"))
        st.markdown("---")
        st.subheader("Notes")
        st.info("Tree-based models (Random Forest & Gradient Boosting) captured behavioral non-linearities. Logistic regression remains a strong linear baseline.")

# function to render model detail panels (bar, scatter, hist)
def render_model_detail(model_name, df_metrics):
    st.subheader(f"{model_name} — Metrics Overview")
    row1, row2 = st.columns([2, 1])
    model_row = df_metrics[df_metrics["Model"] == model_name].iloc[0]
    with row1:
        cols = st.columns(4)
        metrics = ["Accuracy", "Precision", "Recall", "F1"]
        for c, m in zip(cols, metrics):
            with c:
                st.markdown(metric_card_html(m, f"{model_row[m]:.1%}" if not np.isnan(model_row[m]) else "N/A"), unsafe_allow_html=True)
        st.divider()
        # Charts
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.plotly_chart(plot_bar_metrics(df_metrics, metric="AUC"), use_container_width=True)
        with chart_col2:
            st.plotly_chart(plot_hist_metric(df_metrics, metric="AUC"), use_container_width=True)
        st.markdown("#### Metric Relationship")
        st.plotly_chart(plot_scatter_metrics(df_metrics, x_metric="Accuracy", y_metric="Recall"), use_container_width=True)

# ------------------------------
# TAB 2: Random Forest
# ------------------------------
with tabs[1]:
    st.header("🌲 Random Forest Analysis")
    render_model_detail("Random Forest", df_compare)
    # Feature importance & confusion matrix (if present in perf_data)
    fi = perf_data.get("feature_importance", None) if perf_data else None
    cm = perf_data.get("confusion_matrix", None) if perf_data else None
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Top Predictors")
        if fi:
            fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())}).sort_values("Importance", ascending=True)
            fig = px.bar(fi_df.tail(12), x="Importance", y="Feature", orientation="h", title="Feature Importance (Random Forest)", color_discrete_sequence=[ACCENT])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not found in `model_performance.json`. Provide it to visualize predictors.")
    with col_b:
        st.markdown("#### Confusion Matrix")
        if cm:
            cm_arr = np.array(cm)
            fig = px.imshow(cm_arr, text_auto=True, color_continuous_scale='Purples', title="Confusion Matrix (RF)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Confusion matrix not available. Add 'confusion_matrix' key to your performance JSON.")

# ------------------------------
# TAB 3: Gradient Boosting
# ------------------------------
with tabs[2]:
    st.header("🚀 Gradient Boosting")
    render_model_detail("Gradient Boosting", df_compare)
    st.write("Gradient Boosting often requires more careful hyperparameter tuning and training time but can closely match RF performance on tabular data.")

# ------------------------------
# TAB 4: Logistic Regression
# ------------------------------
with tabs[3]:
    st.header("📈 Logistic Regression")
    render_model_detail("Logistic Regression", df_compare)
    st.write("Logistic Regression indicates strong linear signals for some ghosting indicators (e.g., response time).")

# ------------------------------
# TAB 5: Ensemble
# ------------------------------
with tabs[4]:
    st.header("🔮 Ensemble (Final Model)")
    render_model_detail("Ensemble", df_compare)
    # ROC if present
    if perf_data and "roc_curve" in perf_data:
        st.markdown("#### ROC Curve (Ensemble)")
        fpr = np.array(perf_data["roc_curve"]["fpr"])
        tpr = np.array(perf_data["roc_curve"]["tpr"])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Ensemble ROC", line=dict(color=ACCENT)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
        fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=420)
        st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.info("ROC curve data not found in `model_performance.json`.")

# ------------------------------
# TAB 6: Interactive Simulator
# ------------------------------
with tabs[5]:
    st.header("🧪 Live Simulator — Predict Risk")
    left, right = st.columns([2, 1])
    with left:
        st.markdown("#### Input Controls")
        msg_count = st.slider("Messages Sent (recent)", min_value=0, max_value=200, value=20)
        emoji_rate = st.slider("Emoji Usage Rate (0-1)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        response_time = st.slider("Avg. Reply Delay (hours)", 0.0, 120.0, 12.0, step=0.5)
        has_history = st.checkbox("Has ghosted others before?", value=False)
        relationship_status = st.selectbox("Relationship status", ["single", "casual", "dating", "complicated"])
        # Additional controls can be added as needed
        st.markdown("#### Advanced (Optional)")
        use_scaler = st.checkbox("Apply scaler (if available)", value=False)
    with right:
        st.markdown("#### Prediction Result")
        if model and model_columns is not None:
            if st.button("Predict Risk", type="primary"):
                # Build input vector with zeros and map known inputs
                input_df = pd.DataFrame(columns=model_columns)
                input_df.loc[0] = 0
                # Map heuristics - adapt column names to your model
                if "Message_Sent_Count" in input_df.columns:
                    input_df.at[0, "Message_Sent_Count"] = msg_count
                if "Emoji_Usage_Rate" in input_df.columns:
                    input_df.at[0, "Emoji_Usage_Rate"] = emoji_rate
                if "Avg_Reply_Delay_Hours" in input_df.columns or "Response_Time" in input_df.columns:
                    col_r = "Avg_Reply_Delay_Hours" if "Avg_Reply_Delay_Hours" in input_df.columns else "Response_Time"
                    input_df.at[0, col_r] = response_time
                if "Has_Ghosting_History" in input_df.columns:
                    input_df.at[0, "Has_Ghosting_History"] = int(has_history)
                # one-hot relationship status if present
                status_col = f"Relationship_{relationship_status}"
                if status_col in input_df.columns:
                    input_df.at[0, status_col] = 1
                # apply scaler if requested
                try:
                    if use_scaler and scaler is not None:
                        input_df[input_df.columns] = scaler.transform(input_df[input_df.columns])
                except Exception:
                    st.warning("Scaler application failed - check scaler compatibility.")

                # predict
                try:
                    if hasattr(model, "predict_proba"):
                        prob = float(model.predict_proba(input_df.values)[0][1])
                    else:
                        # fallback to predict output
                        pred = model.predict(input_df.values)[0]
                        prob = float(pred)
                    st.metric("Ghosting Probability", f"{prob:.1%}")
                    if prob >= 0.7:
                        st.error("⚠️ High Risk: User likely to ghost")
                    elif prob >= 0.4:
                        st.warning("⚠️ Medium Risk: Keep an eye on interaction")
                    else:
                        st.success("✅ Low Risk: Likely to respond")
                except Exception as e:
                    st.error(f"Model inference error: {e}")
        else:
            st.warning("Model files not loaded. Ensure `ghosting_risk_model.pkl` and `model_columns.pkl` exist or set your GDRIVE_MODEL_ID environment variable.")

    st.markdown("---")
    st.markdown("#### Simulation Insights")
    st.write("You can use the sliders to test hypothetical user reply behavior and observe how the model responds. Export your `model_performance.json` with keys: `feature_importance`, `confusion_matrix`, `roc_curve` to populate advanced charts.")

# --- FOOTER ---
st.markdown("---")
st.markdown("<div class='small-muted'>Tip: to fully enable the live simulator, provide `ghosting_risk_model.pkl` and `model_columns.pkl` (joblib). You can export `model_performance.json` from your notebook to show feature importance, confusion matrix & ROC.</div>", unsafe_allow_html=True)
