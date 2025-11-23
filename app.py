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
# Ghosting Research — Modern Streamlit Dashboard
# Single-file app: Ghosting_Dashboard_Modern.py
# -----------------------------------------------

# --- PAGE CONFIG ---
st.set_page_config(page_title="Ghosting Research — Modern Dashboard", layout="wide", initial_sidebar_state="collapsed")

# --- STYLES (Modern Purple Theme + Card Design) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    /* Page background */
    .reportview-container, .main, .block-container { background-color: #f7f7fb; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #5a00d4 0%, #b47bff 100%);
        padding: 28px 22px;
        border-radius: 18px;
        color: white;
        box-shadow: 0 12px 30px rgba(90,0,212,0.16);
        margin-bottom: 18px;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-weight: 700; font-size: 26px; }
    .main-header p { margin: 4px 0 0; opacity: 0.92; }

    /* Metric Card */
    .metric-card { background: #fff; border-radius: 14px; padding: 18px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); border: 1px solid #f0edf8; }
    .metric-label { color: #6b6b80; font-weight: 600; font-size: 0.85rem; }
    .metric-value { font-size: 1.9rem; font-weight: 700; color: #272343; margin-top: 6px; }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs button { border-radius: 12px !important; padding: 8px 16px !important; }
    .stTabs button[aria-selected="true"] { background: linear-gradient(90deg,#6b21a8,#8b5cf6) !important; color: white !important; }

    /* Small utility */
    .muted { color: #7b7b93; font-size: 0.9rem; }

    /* Sidebar */
    .streamlit-expanderHeader { font-weight: 600; }

</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="main-header"><h1>👻 Ghosting Prediction Research</h1><p class="muted">Behavioral analysis • ML evaluation • Interactive simulator</p></div>', unsafe_allow_html=True)

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    # Adjust file IDs / paths as needed. This function will attempt to load:
    # - model_performance.json
    # - model pickle (ghosting_risk_model.pkl)
    # - model_columns.pkl
    # - scaler.pkl
    file_id = '1gAogfnZDcpuSOTLa0UTD4tvXM0Vk0Jp2'  # set if you want gdown fallback
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

    return perf_data, model, columns, scaler

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

# --- MAIN TABS ---
tabs = st.tabs(["🏆 Overview", "🌲 Random Forest", "🚀 Gradient Boosting", "🔮 Ensemble", "🧪 Simulator", "📁 Data & Downloads"]) 

# ------------------ TAB: OVERVIEW ------------------
with tabs[0]:
    st.subheader("Model Leaderboard & Key Insights")

    # Leaderboard
    df_compare = pd.DataFrame.from_dict(all_models_data, orient='index').reset_index()
    df_compare.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    df_compare = df_compare.sort_values(by='AUC', ascending=False)

    left, right = st.columns([2,1])

    with left:
        fig = px.bar(df_compare, x='AUC', y='Model', orientation='h', color='AUC', color_continuous_scale='Purples', text_auto='.4f')
        fig.update_layout(title='Model Comparison by AUC (Higher is Better)', height=420, margin=dict(l=40,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Primary Metric</div>', unsafe_allow_html=True)
        top = df_compare.iloc[0]
        st.markdown(f'<div class="metric-value">{top.AUC:.4f} AUC</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Top model: <b>{}</b></div>'.format(top.Model), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('---')
    st.subheader('Accuracy vs Recall (Trade-offs)')
    fig_scatter = px.scatter(df_compare, x='Recall', y='Accuracy', size='AUC', color='Model', hover_name='Model', size_max=60)
    st.plotly_chart(fig_scatter, use_container_width=True)

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

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">AUC</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{metrics["AUC"]:.4f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        df_metrics = pd.DataFrame({
            'Metric':['Accuracy','Precision','Recall','F1','AUC'],
            'Score':[metrics['Accuracy'], metrics['Precision'], metrics['Recall'], metrics['F1'], metrics['AUC']]
        })
        fig_bar = px.bar(df_metrics, x='Metric', y='Score', color='Score', color_continuous_scale='Greens', text_auto='.2%')
        fig_bar.update_layout(yaxis_range=[0,1.05], height=340)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Feature importance (if available)
    st.subheader('Top Predictors (Feature Importance)')
    fi = perf_data.get('feature_importance') if perf_data else None
    if fi:
        fi_df = pd.DataFrame({'Feature':list(fi.keys()), 'Importance':list(fi.values())}).sort_values('Importance', ascending=True).tail(12)
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title='Top Features — Random Forest', text_auto='.2f')
        st.plotly_chart(fig_fi, use_container_width=True)
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
            if label=='AUC': st.markdown(f'<div class="metric-value">{val:.4f}</div>', unsafe_allow_html=True)
            else: st.markdown(f'<div class="metric-value">{pct(val)}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Performance breakdown
    df_metrics = pd.DataFrame({
        'Metric':['Accuracy','Precision','Recall','F1','AUC'],
        'Score':[metrics['Accuracy'], metrics['Precision'], metrics['Recall'], metrics['F1'], metrics['AUC']]
    })
    fig = px.line(df_metrics, x='Metric', y='Score', markers=True, title='Metric Trend (GB)')
    fig.update_yaxes(range=[0,1.05])
    st.plotly_chart(fig, use_container_width=True)

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
            if label=='AUC': st.markdown(f'<div class="metric-value">{val:.4f}</div>', unsafe_allow_html=True)
            else: st.markdown(f'<div class="metric-value">{pct(val)}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('---')

    # ROC Curve (if available)
    st.subheader('ROC Curve')
    roc = perf_data.get('roc_curve')
    if roc and 'fpr' in roc and 'tpr' in roc:
        fpr, tpr = roc['fpr'], roc['tpr']
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
        fig_roc.add_shape(type='line', x0=0, x1=1, y0=0, y1=1, line=dict(dash='dash'))
        fig_roc.update_layout(title='Ensemble ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=420)
        st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.info('ROC data not found in `model_performance.json`. Add `roc_curve` with fpr & tpr arrays to visualize.')

    # Confusion Matrix
    st.subheader('Confusion Matrix')
    cm = perf_data.get('confusion_matrix')
    if cm:
        cm = np.array(cm)
        labels_x = ['Pred: No Ghost','Pred: Ghost']
        labels_y = ['Actual: No Ghost','Actual: Ghost']
        fig_cm = go.Figure(data=go.Heatmap(z=cm, x=labels_x, y=labels_y, text=cm, texttemplate='%{text}', colorscale='Purples'))
        fig_cm.update_layout(height=360, margin=dict(l=40,r=20,t=40,b=20))
        st.plotly_chart(fig_cm, use_container_width=True)
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
            demo_prob = min(0.98, demo_prob)
            fig_gauge = go.Figure(go.Indicator(mode='gauge+number', value=demo_prob*100, title={'text':'Ghosting Probability'}, gauge={'axis':{'range':[0,100]}, 'steps':[{'range':[0,50],'color':'#b7f0c1'},{'range':[50,100],'color':'#ffd6d6'}], 'bar':{'color':'#3b1f6b'}}))
            st.plotly_chart(fig_gauge, use_container_width=True)
            if demo_prob > 0.5:
                st.error('RESULT: High Risk Conversation (Demo)')
            else:
                st.success('RESULT: Low Risk Conversation (Demo)')
        else:
            if st.button('Run Model'):
                # Build input vector
                input_df = pd.DataFrame(columns=model_columns)
                input_df.loc[0] = 0
                # safe assignments
                if 'Message_Sent_Count' in input_df.columns: input_df['Message_Sent_Count'] = msg_count
                if 'Emoji_Usage_Rate' in input_df.columns: input_df['Emoji_Usage_Rate'] = emoji_rate
                if 'Avg_Response_Time_Hours' in input_df.columns: input_df['Avg_Response_Time_Hours'] = response_time
                if 'Has_Ghosting_History' in input_df.columns: input_df['Has_Ghosting_History'] = 1 if has_history else 0

                try:
                    if scaler is not None:
                        X = scaler.transform(input_df)
                    else:
                        X = input_df.values
                    prob = model.predict_proba(X)[0][1]

                    fig_gauge = go.Figure(go.Indicator(mode='gauge+number', value=prob*100, title={'text':'Ghosting Probability'}, gauge={'axis':{'range':[0,100]}, 'steps':[{'range':[0,50],'color':'#b7f0c1'},{'range':[50,100],'color':'#ffd6d6'}], 'bar':{'color':'#3b1f6b'}}))
                    st.plotly_chart(fig_gauge, use_container_width=True)

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
        files = [f for f in os.listdir('.') if f.endswith(('.csv', '.json', '.pkl'))]
        for f in files:
            st.write(f)
            try:
                st.download_button(label=f'Download {f}', data=open(f,'rb'), file_name=f)
            except Exception:
                pass

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
st.markdown('<div style="padding:12px 0; opacity:0.7; font-size:0.9rem">Made with ❤️ — Ghosting Research Dashboard. Need layout tweaks? Tell me exactly what to change.</div>', unsafe_allow_html=True)
