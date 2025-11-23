# Ghosting_Dashboard_Modern.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import gdown

# -----------------------------------------------
# Ghosting Research — Modern Streamlit Dashboard (Theme A: Dark Purple Neon)
# Single-file app
# -----------------------------------------------

# --- PAGE CONFIG ---
st.set_page_config(page_title="Ghosting Research — Modern Dashboard", layout="wide", initial_sidebar_state="collapsed")

# --- DARK PURPLE NEON THEME (Theme A) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

/* GLOBAL BACKGROUND */
.reportview-container, .main, .block-container {
    background: linear-gradient(180deg, #0d021f 0%, #090015 100%) !important;
}

/* HEADER */
.main-header {
    background: linear-gradient(135deg, #5a00d4 0%, #b47bff 100%);
    padding: 28px 22px;
    border-radius: 16px;
    color: white;
    box-shadow: 0 14px 40px rgba(90,0,212,0.28);
    margin-bottom: 18px;
    text-align: center;
}
.main-header h1 { margin: 0; font-weight: 700; font-size: 28px; letter-spacing: 0.2px; }
.main-header p { margin: 4px 0 0; opacity: 0.95; color: #efe7ff; }

/* Metric Card */
.metric-card {
    background: linear-gradient(180deg, #120428 0%, #190534 100%);
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 8px 26px rgba(0,0,0,0.6);
    border: 1px solid rgba(139,92,246,0.14);
}
.metric-label { color: #c9bfe8; font-weight: 600; font-size: 0.88rem; }
.metric-value { font-size: 1.95rem; font-weight: 700; color: #ffffff; margin-top: 6px; }

/* Tabs -> dark neon buttons */
.stTabs [data-baseweb="tab-list"] { gap: 12px; margin-bottom: 8px; }
.stTabs button {
    background: linear-gradient(180deg,#110426,#19052b) !important;
    color: #d8c7ff !important;
    border-radius: 12px !important;
    padding: 10px 18px !important;
    border: 1px solid rgba(110,40,200,0.18) !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.6);
}
.stTabs button[aria-selected="true"] {
    background: linear-gradient(90deg, #6b21a8, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
}

/* Muted text */
.muted { color: #bda9d6; font-size: 0.95rem; }

/* Inputs & small utility */
.stSlider > div, .stNumberInput > div, .stTextInput > div {
    background: transparent !important;
}
.streamlit-expanderHeader { font-weight: 600; color: #f3eaff !important; }

/* Remove white boxes around widgets where possible */
.widget-label, label {
    color: #efe7ff !important;
}

/* Footer small text */
.footer-small { color: #d6c9f3; opacity:0.8; padding-top: 8px; }

/* Make download list text lighter */
div[data-testid="stFileUploader"] > label { color: #e8ddff !important; }

</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="main-header"><h1>👻 Ghosting Prediction Research</h1><p class="muted">Behavioral analysis • ML evaluation • Interactive simulator</p></div>', unsafe_allow_html=True)

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    # Adjust file IDs / paths as needed.
    file_id = '1gAogfnZDcpuSOTLa0UTD4tvXM0Vk0Jp2'  # optionally set a Google Drive file id for model fallback
    model_filename = 'ghosting_risk_model.pkl'

    # try download model if missing (non-fatal)
    if not os.path.exists(model_filename) and file_id:
        try:
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_filename, quiet=True)
        except Exception:
            pass

    perf_data = None
    model, columns, scaler = None, None, None
    try:
        if os.path.exists('model_performance.json'):
            with open('model_performance.json', 'r') as f:
                perf_data = json.load(f)
    except Exception:
        perf_data = None

    try:
        if os.path.exists(model_filename):
            model = joblib.load(model_filename)
        if os.path.exists('model_columns.pkl'):
            columns = joblib.load('model_columns.pkl')
        if os.path.exists('scaler.pkl'):
            scaler = joblib.load('scaler.pkl')
    except Exception:
        model, columns, scaler = None, None, None

    return perf_data or {}, model, columns, scaler

perf_data, model, model_columns, scaler = load_assets()

# If perf_data missing, provide a safe fallback so layout still shows
if not perf_data:
    st.warning("Model performance file not found. The app will show demo metrics and visuals. Add 'model_performance.json' for full data.")
    perf_data = {}

# --- SAMPLE/DEFAULT METRICS (fallback) ---
all_models_data = {
    "Random Forest": {"Accuracy": 0.9352, "Precision": 0.89, "Recall": 0.87, "F1": 0.88, "AUC": 0.9352},
    "Ensemble": {"Accuracy": 0.9337, "Precision": 0.89, "Recall": 0.86, "F1": 0.87, "AUC": 0.9337},
    "Gradient Boosting": {"Accuracy": 0.9319, "Precision": 0.89, "Recall": 0.86, "F1": 0.87, "AUC": 0.9319},
    "Logistic Regression": {"Accuracy": 0.8953, "Precision": 0.92, "Recall": 0.84, "F1": 0.88, "AUC": 0.8953},
}

# --- UTIL: format percent ---
def pct(x):
    try:
        return f"{x:.1%}"
    except Exception:
        return str(x)

# --- HELPER: modern_plot_fig (dark neon style, bigger) ---
def modern_plot_fig(fig, height=520):
    fig.update_layout(
        template="plotly_dark",
        font=dict(family="Poppins", size=14, color="#f2e9ff"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=60, b=40),
        height=height
    )
    fig.update_xaxes(showgrid=True, gridcolor="#2b1740", zerolinecolor="#2b1740", tickfont=dict(color="#e6dbff"))
    fig.update_yaxes(showgrid=True, gridcolor="#2b1740", zerolinecolor="#2b1740", tickfont=dict(color="#e6dbff"))
    return fig

# --- MAIN TABS ---
tabs = st.tabs(["🏆 Overview", "🌲 Random Forest", "🚀 Gradient Boosting", "🔮 Ensemble", "🧪 Simulator", "📁 Data & Downloads"])

# ------------------ TAB: OVERVIEW ------------------
with tabs[0]:
    st.subheader("Model Leaderboard & Key Insights")
    df_compare = pd.DataFrame.from_dict(all_models_data, orient='index').reset_index()
    df_compare.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    df_compare = df_compare.sort_values(by='AUC', ascending=False)

    left, right = st.columns([2,1])

    with left:
        fig = px.bar(df_compare, x='AUC', y='Model', orientation='h', color='AUC', color_continuous_scale=[[0, "#7b64d9"], [1, "#b47bff"]], text='AUC')
        fig.update_traces(texttemplate='%{text:.4f}', textposition='inside', marker_line_width=0)
        fig.update_layout(title='Model Comparison by AUC (Higher is Better)', height=520, margin=dict(l=40,r=20,t=60,b=20))
        st.plotly_chart(modern_plot_fig(fig), use_container_width=True)

    with right:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Primary Metric</div>', unsafe_allow_html=True)
        top = df_compare.iloc[0]
        st.markdown(f'<div class="metric-value">{top.AUC:.4f} AUC</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="muted">Top model: <b>{top.Model}</b></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('---')
    st.subheader('Accuracy vs Recall (Trade-offs)')
    fig_scatter = px.scatter(df_compare, x='Recall', y='Accuracy', size='AUC', color='Model', hover_name='Model', size_max=90)
    fig_scatter.update_layout(height=520, legend_title=None)
    st.plotly_chart(modern_plot_fig(fig_scatter), use_container_width=True)

# ------------------ TAB: RANDOM FOREST ------------------
with tabs[1]:
    st.subheader('🌲 Random Forest — Performance')
    metrics = all_models_data['Random Forest']

    c1, c2 = st.columns([1,2])
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Accuracy</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{pct(metrics["Accuracy"])}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">AUC</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{metrics["AUC"]:.4f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        df_metrics = pd.DataFrame({
            'Metric':['Accuracy','Precision','Recall','F1','AUC'],
            'Score':[metrics['Accuracy'], metrics['Precision'], metrics['Recall'], metrics['F1'], metrics['AUC']]
        })
        fig_bar = px.bar(df_metrics, x='Metric', y='Score', color='Score', color_continuous_scale=[[0, "#4f2a7a"], [1, "#b47bff"]], text_auto='.2%')
        fig_bar.update_layout(yaxis_range=[0,1.05], height=520)
        st.plotly_chart(modern_plot_fig(fig_bar), use_container_width=True)

    # Feature importance (if available)
    st.subheader('Top Predictors (Feature Importance)')
    fi = perf_data.get('feature_importance') if perf_data else None
    if fi:
        fi_df = pd.DataFrame({'Feature':list(fi.keys()), 'Importance':list(fi.values())}).sort_values('Importance', ascending=True).tail(12)
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title='Top Features — Random Forest', text_auto='.2f')
        st.plotly_chart(modern_plot_fig(fig_fi), use_container_width=True)
    else:
        st.info('Feature importance not found in `model_performance.json`. Add it to display top predictors.')

# ------------------ TAB: GRADIENT BOOSTING ------------------
with tabs[2]:
    st.subheader('🚀 Gradient Boosting — Performance')
    metrics = all_models_data['Gradient Boosting']
    show_cols = st.columns(4)
    for idx, (label, key) in enumerate([('Accuracy','Accuracy'), ('Precision','Precision'), ('Recall','Recall'), ('AUC','AUC')]):
        with show_cols[idx]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">{label}</div>', unsafe_allow_html=True)
            val = metrics[key]
            if label=='AUC':
                st.markdown(f'<div class="metric-value">{val:.4f}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-value">{pct(val)}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    df_metrics = pd.DataFrame({
        'Metric':['Accuracy','Precision','Recall','F1','AUC'],
        'Score':[metrics['Accuracy'], metrics['Precision'], metrics['Recall'], metrics['F1'], metrics['AUC']]
    })
    fig = px.line(df_metrics, x='Metric', y='Score', markers=True)
    fig.update_layout(title='Metric Trend (Gradient Boosting)', height=520)
    st.plotly_chart(modern_plot_fig(fig), use_container_width=True)

# ------------------ TAB: ENSEMBLE ------------------
with tabs[3]:
    st.subheader('🔮 Voting Ensemble (Final Model) — Evaluation')
    metrics = all_models_data['Ensemble']
    show_cols = st.columns(4)
    for idx, (label, key) in enumerate([('Accuracy','Accuracy'), ('Precision','Precision'), ('Recall','Recall'), ('AUC','AUC')]):
        with show_cols[idx]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">{label}</div>', unsafe_allow_html=True)
            val = metrics[key]
            if label=='AUC':
                st.markdown(f'<div class="metric-value">{val:.4f}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-value">{pct(val)}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('---')

    # ROC Curve (if available)
    st.subheader('ROC Curve')
    roc = perf_data.get('roc_curve') or perf_data.get('roc') or {}
    # Accept different key naming possibilities
    fpr = roc.get('fpr') or perf_data.get('fpr')
    tpr = roc.get('tpr') or perf_data.get('tpr')
    if fpr is not None and tpr is not None:
        try:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC', line=dict(width=3)))
            fig_roc.add_shape(type='line', x0=0, x1=1, y0=0, y1=1, line=dict(dash='dash', color='rgba(255,255,255,0.12)'))
            fig_roc.update_layout(title='Ensemble ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=520)
            st.plotly_chart(modern_plot_fig(fig_roc), use_container_width=True)
        except Exception:
            st.info('Provided ROC arrays could not be plotted (invalid format).')
    else:
        st.info('ROC data not found in `model_performance.json`. Add `roc_curve` with fpr & tpr arrays to visualize.')

    # Confusion Matrix
    st.subheader('Confusion Matrix')
    cm = perf_data.get('confusion_matrix') or perf_data.get('cm')
    if cm:
        try:
            cm = np.array(cm)
            labels_x = ['Pred: No Ghost','Pred: Ghost']
            labels_y = ['Actual: No Ghost','Actual: Ghost']
            cm_fig = go.Figure(data=go.Heatmap(
                z=cm,
                text=cm,
                texttemplate="%{text}",
                x=labels_x,
                y=labels_y,
                colorscale=[[0, '#1a0633'], [0.4, '#551a8b'], [1, '#b47bff']],
                hoverongaps=False,
                showscale=False
            ))
            cm_fig.update_layout(height=480, margin=dict(l=40,r=20,t=40,b=20))
            st.plotly_chart(modern_plot_fig(cm_fig, height=480), use_container_width=True)
        except Exception:
            st.info('Confusion matrix found but could not render it. Ensure it is a 2x2 numeric list.')
    else:
        st.info('Confusion matrix not present in performance file. Add `confusion_matrix` key (2x2 list) to display.')

# ------------------ TAB: SIMULATOR ------------------
with tabs[4]:
    st.subheader('🧪 Interactive Risk Simulator')

    sim_left, sim_right = st.columns([1,2])
    with sim_left:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Simulation Inputs</div>', unsafe_allow_html=True)
        msg_count = st.slider('Messages Sent', 0, 200, 20)
        emoji_rate = st.slider('Emoji Usage Rate (0-1)', 0.0, 1.0, 0.1, step=0.05)
        response_time = st.slider('Average Response Time (hrs)', 0.0, 72.0, 12.0)
        has_history = st.checkbox('Has Ghosting History?', value=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with sim_right:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Prediction</div>', unsafe_allow_html=True)

        if not model or not model_columns:
            st.info('Model files not loaded. Use local model files (ghosting_risk_model.pkl, model_columns.pkl) to enable live predictions.')
            # Provide a simulated probability for demo
            demo_prob = 0.35 + (msg_count/200)*0.3 + (emoji_rate*0.2) + (0.15 if has_history else 0)
            demo_prob = float(min(0.98, demo_prob))
            fig_gauge = go.Figure(go.Indicator(
                mode='gauge+number',
                value=demo_prob*100,
                title={'text':'Ghosting Probability'},
                gauge={
                    'axis':{'range':[0,100]},
                    'steps':[{'range':[0,50],'color':'#2e7d32'},{'range':[50,100],'color':'#a60f2a'}],
                    'bar':{'color':'#b47bff'}
                }
            ))
            st.plotly_chart(modern_plot_fig(fig_gauge, height=380), use_container_width=True)
            if demo_prob > 0.5:
                st.error('RESULT: High Risk Conversation (Demo)')
            else:
                st.success('RESULT: Low Risk Conversation (Demo)')
        else:
            if st.button('Run Model'):
                # Build input vector
                try:
                    input_df = pd.DataFrame(columns=model_columns)
                    input_df.loc[0] = 0
                    # safe assignments
                    if 'Message_Sent_Count' in input_df.columns: input_df['Message_Sent_Count'] = int(msg_count)
                    if 'Emoji_Usage_Rate' in input_df.columns: input_df['Emoji_Usage_Rate'] = float(emoji_rate)
                    if 'Avg_Response_Time_Hours' in input_df.columns: input_df['Avg_Response_Time_Hours'] = float(response_time)
                    if 'Has_Ghosting_History' in input_df.columns: input_df['Has_Ghosting_History'] = 1 if has_history else 0

                    if scaler is not None:
                        X = scaler.transform(input_df)
                    else:
                        X = input_df.values
                    prob = float(model.predict_proba(X)[0][1])

                    fig_gauge = go.Figure(go.Indicator(
                        mode='gauge+number',
                        value=prob*100,
                        title={'text':'Ghosting Probability'},
                        gauge={
                            'axis':{'range':[0,100]},
                            'steps':[{'range':[0,50],'color':'#2e7d32'},{'range':[50,100],'color':'#a60f2a'}],
                            'bar':{'color':'#b47bff'}
                        }
                    ))
                    st.plotly_chart(modern_plot_fig(fig_gauge, height=380), use_container_width=True)

                    if prob > 0.5:
                        st.error('RESULT: High Risk Conversation')
                    else:
                        st.success('RESULT: Low Risk Conversation')
                except Exception as e:
                    st.error(f'Prediction Error: {e}')

        st.markdown('</div>', unsafe_allow_html=True)

# ------------------ TAB: DATA & DOWNLOADS ------------------
with tabs[5]:
    st.subheader('📁 Data, Model Files & Downloads')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**Available files (local)**')
        files = sorted([f for f in os.listdir('.') if f.endswith(('.csv', '.json', '.pkl'))])
        if files:
            for f in files:
                st.write(f)
                try:
                    with open(f,'rb') as file_obj:
                        st.download_button(label=f'Download {f}', data=file_obj, file_name=f)
                except Exception:
                    st.write('Unable to make download button for', f)
        else:
            st.info("No local .csv/.json/.pkl files found in working directory.")

    with col2:
        st.markdown('**Upload your model_performance.json**')
        uploaded = st.file_uploader('Upload JSON', type=['json'])
        if uploaded:
            try:
                data = json.load(uploaded)
                with open('model_performance.json', 'w') as out:
                    json.dump(data, out)
                st.success('Saved model_performance.json — reload the app to use it')
            except Exception as e:
                st.error(f'Invalid JSON: {e}')

# ------------------ FOOTER ------------------
st.markdown('<div class="footer-small">Made with ❤️ — Ghosting Research Dashboard. Need layout tweaks? Tell me exactly what to change.</div>', unsafe_allow_html=True)

# ------------------ EXTRA: Enhanced Confusion Matrix & ROC if present (global bottom area) ------------------
# Render enhanced visuals if the perf_data contains keys with common names
if perf_data:
    # Confusion matrix variations
    cm_keys = perf_data.get('confusion_matrix') or perf_data.get('confusion') or perf_data.get('cm')
    if cm_keys:
        try:
            cm_arr = np.array(cm_keys)
            st.markdown('---')
            st.subheader("Confusion Matrix (Enhanced)")
            cm_fig = go.Figure(data=go.Heatmap(
                z=cm_arr,
                text=cm_arr,
                texttemplate="%{text}",
                x=["Predicted No Ghost", "Predicted Ghost"],
                y=["Actual No Ghost", "Actual Ghost"],
                colorscale=[[0, '#1a0633'], [0.4, '#551a8b'], [1, '#b47bff']],
                showscale=False
            ))
            st.plotly_chart(modern_plot_fig(cm_fig, height=420), use_container_width=True)
        except Exception:
            pass

    # ROC variations
    fpr = perf_data.get('fpr') or perf_data.get('roc_curve', {}).get('fpr')
    tpr = perf_data.get('tpr') or perf_data.get('roc_curve', {}).get('tpr')
    if fpr is not None and tpr is not None:
        try:
            st.markdown('---')
            st.subheader("ROC Curve (Enhanced)")
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC', line=dict(width=3)))
            roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Baseline', line=dict(dash='dash', color='rgba(255,255,255,0.12)')))
            st.plotly_chart(modern_plot_fig(roc_fig, height=420), use_container_width=True)
        except Exception:
            pass
