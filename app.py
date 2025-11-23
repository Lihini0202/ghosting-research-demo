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
# Ghosting Research — Modern Streamlit Dashboard (Fixed Formatting)
# -----------------------------------------------

# --- PAGE CONFIG ---
st.set_page_config(page_title="Ghosting Research — Dashboard", layout="wide", initial_sidebar_state="collapsed")

# --- STYLES (Dark Purple Neon Theme + Card Design) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; background-color: #1a0a3c; color: #f0f0f0; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #5a00d4 0%, #b47bff 100%);
        padding: 22px 18px;
        border-radius: 12px;
        color: white;
        box-shadow: 0 8px 20px rgba(90,0,212,0.25);
        margin-bottom: 12px;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-weight: 700; font-size: 24px; }
    .main-header p { margin: 4px 0 0; opacity: 0.85; }

    /* Metric Card */
    .metric-card { background: #2c1b5c; border-radius: 12px; padding: 14px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); border: 1px solid #4b2fa6; }
    .metric-label { color: #c8c8e2; font-weight: 500; font-size: 0.85rem; }
    .metric-value { font-size: 1.7rem; font-weight: 700; color: #fff; margin-top: 6px; }

    /* Small utility */
    .muted { color: #b0b0c5; font-size: 0.9rem; }

    /* Sidebar */
    .streamlit-expanderHeader { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="main-header"><h1>👻 Ghosting Prediction Research</h1><p class="muted">Behavioral analysis • ML evaluation • Interactive simulator</p></div>', unsafe_allow_html=True)

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    file_id = '1gAogfnZDcpuSOTLa0UTD4tvXM0Vk0Jp2'
    model_filename = 'ghosting_risk_model.pkl'
    if not os.path.exists(model_filename) and file_id:
        try:
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_filename, quiet=True)
        except: pass

    perf_data = None
    model, columns, scaler = None, None, None
    try:
        if os.path.exists('model_performance.json'):
            with open('model_performance.json','r') as f:
                perf_data = json.load(f)
    except: perf_data = None

    try:
        if os.path.exists(model_filename):
            model = joblib.load(model_filename)
        if os.path.exists('model_columns.pkl'):
            columns = joblib.load('model_columns.pkl')
        if os.path.exists('scaler.pkl'):
            scaler = joblib.load('scaler.pkl')
    except: model, columns, scaler = None, None, None

    return perf_data, model, columns, scaler

perf_data, model, model_columns, scaler = load_assets()
if not perf_data:
    st.warning("Model performance file missing. Showing demo metrics.")
    perf_data = {}

# --- SAMPLE METRICS ---
all_models_data = {
    "Random Forest": {"Accuracy": 0.9352, "Precision": 0.89, "Recall": 0.87, "F1": 0.88, "AUC": 0.9352},
    "Ensemble": {"Accuracy": 0.9337, "Precision": 0.89, "Recall": 0.86, "F1": 0.87, "AUC": 0.9337},
    "Gradient Boosting": {"Accuracy": 0.9319, "Precision": 0.89, "Recall": 0.86, "F1": 0.87, "AUC": 0.9319},
    "Logistic Regression": {"Accuracy": 0.8953, "Precision": 0.92, "Recall": 0.84, "F1": 0.88, "AUC": 0.8953},
}

def pct(x): return f"{x:.1%}" if isinstance(x,float) else str(x)

# --- MAIN TABS ---
tabs = st.tabs(["🏆 Overview","🌲 Random Forest","🚀 Gradient Boosting","🔮 Ensemble","🧪 Simulator","📁 Data & Downloads"])

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
        fig.update_layout(title='Model Comparison by AUC (Higher is Better)', height=300, margin=dict(l=40,r=20,t=50,b=20))
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

    # Scatter plot with smaller markers
    fig_scatter = px.scatter(
        df_compare,
        x='Recall',
        y='Accuracy',
        size_max=15,  # maximum size of dots
        size=[10]*len(df_compare),  # uniform smaller size
        color='Model',
        hover_name='Model'
    )
    fig_scatter.update_traces(marker=dict(size=10))  # enforce smaller dot size
    fig_scatter.update_layout(height=300, margin=dict(l=40,r=20,t=40,b=20))  # compact height
    st.plotly_chart(fig_scatter, use_container_width=True)


# ------------------ TAB: RANDOM FOREST ------------------
with tabs[1]:
    st.subheader('🌲 Random Forest — Performance')
    metrics = all_models_data['Random Forest']
    c1, c2 = st.columns([1,2])
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Accuracy</div>', unsafe_allow_html=True)
        display_val = pct(metrics["Accuracy"])
        st.markdown(f'<div class="metric-value">{display_val}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">AUC</div>', unsafe_allow_html=True)
        display_val = f"{metrics['AUC']:.4f}"
        st.markdown(f'<div class="metric-value">{display_val}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        df_metrics = pd.DataFrame({'Metric':['Accuracy','Precision','Recall','F1','AUC'],
                                   'Score':[metrics['Accuracy'], metrics['Precision'], metrics['Recall'], metrics['F1'], metrics['AUC']]})
        fig_bar = px.bar(df_metrics, x='Metric', y='Score', color='Score', color_continuous_scale='Viridis', text_auto='.2%')
        fig_bar.update_layout(yaxis_range=[0,1.05], height=300, margin=dict(l=30,r=20,t=30,b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader('Top Predictors (Feature Importance)')
    fi = perf_data.get('feature_importance') if perf_data else None
    if fi:
        fi_df = pd.DataFrame({'Feature':list(fi.keys()), 'Importance':list(fi.values())}).sort_values('Importance', ascending=True).tail(12)
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', text_auto='.2f', color='Importance', color_continuous_scale='Plasma')
        fig_fi.update_layout(height=300, margin=dict(l=30,r=20,t=30,b=20))
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info('Feature importance not available.')

# ------------------ TAB: GRADIENT BOOSTING ------------------
with tabs[2]:
    st.subheader('🚀 Gradient Boosting — Performance')
    metrics = all_models_data['Gradient Boosting']

    # Display main metrics in small cards
    show_cols = st.columns(4)
    for idx, (label, key) in enumerate([('Accuracy','Accuracy'),('Precision','Precision'),('Recall','Recall'),('AUC','AUC')]):
        with show_cols[idx]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">{label}</div>', unsafe_allow_html=True)
            val = metrics[key]
            display_val = f"{val:.4f}" if label=="AUC" else pct(val)
            st.markdown(f'<div class="metric-value">{display_val}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('---')

    # Two-column visualizations: Metrics Histogram & Feature Importance
    col_metrics, col_fi = st.columns(2)

    # Metrics Histogram
    with col_metrics:
        df_metrics = pd.DataFrame({
            'Metric':['Accuracy','Precision','Recall','F1','AUC'],
            'Score':[metrics['Accuracy'], metrics['Precision'], metrics['Recall'], metrics['F1'], metrics['AUC']]
        })
        fig_hist = px.bar(df_metrics, x='Metric', y='Score', color='Score', color_continuous_scale='Blues', text_auto='.2%')
        fig_hist.update_layout(height=300, yaxis_range=[0,1.05], margin=dict(l=30,r=20,t=30,b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    # Feature Importance (if available)
    with col_fi:
        fi = perf_data.get('feature_importance')
        if fi:
            fi_df = pd.DataFrame({'Feature':list(fi.keys()), 'Importance':list(fi.values())}).sort_values('Importance', ascending=True).tail(12)
            fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title='Top Features — Gradient Boosting', text_auto='.2f')
            fig_fi.update_layout(height=300, margin=dict(l=30,r=20,t=30,b=20))
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info('Feature importance not found in `model_performance.json`.')



# ------------------ TAB: ENSEMBLE ------------------
with tabs[3]:
    st.subheader('🔮 Ensemble Model — Evaluation')
    metrics = all_models_data['Ensemble']
    show_cols = st.columns(4)
    for idx, (label,key) in enumerate([('Accuracy','Accuracy'),('Precision','Precision'),('Recall','Recall'),('AUC','AUC')]):
        with show_cols[idx]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">{label}</div>', unsafe_allow_html=True)
            val = metrics[key]
            display_val = f"{val:.4f}" if label=="AUC" else pct(val)
            st.markdown(f'<div class="metric-value">{display_val}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('---')

    # Two-column visualization for ROC and Confusion Matrix
    col_roc, col_cm = st.columns(2)

    # ROC Curve
    with col_roc:
        st.subheader('ROC Curve')
        roc = perf_data.get('roc_curve')
        if roc and 'fpr' in roc and 'tpr' in roc:
            fpr, tpr = roc['fpr'], roc['tpr']
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC', line=dict(width=2)))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash')))
            fig_roc.update_layout(height=300, margin=dict(l=30,r=20,t=30,b=20))
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info('ROC data not available.')

    # Confusion Matrix
    with col_cm:
        st.subheader('Confusion Matrix')
        cm = perf_data.get('confusion_matrix')
        if cm:
            cm_fig = go.Figure(go.Heatmap(
                z=np.array(cm),
                x=["Pred No Ghost","Pred Ghost"],
                y=["Actual No Ghost","Actual Ghost"],
                colorscale=[[0,'#f3eaff'],[1,'#6a00ff']],
                text=np.array(cm),
                texttemplate="%{text}"
            ))
            cm_fig.update_layout(height=300, margin=dict(l=30,r=20,t=30,b=20))
            st.plotly_chart(cm_fig, use_container_width=True)
        else:
            st.info('Confusion matrix not available.')


# ------------------ TAB: SIMULATOR ------------------
with tabs[4]:
    st.subheader('🧪 Interactive Risk Simulator')
    sim_left, sim_right = st.columns([1,2])
    with sim_left:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        msg_count = st.slider('Messages Sent', 0, 200, 20)
        emoji_rate = st.slider('Emoji Usage Rate', 0.0,1.0,0.1,step=0.05)
        response_time = st.slider('Avg Response Time (hrs)',0.0,72.0,12.0)
        has_history = st.checkbox('Has Ghosting History?', value=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with sim_right:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Prediction</div>', unsafe_allow_html=True)
        demo_prob = 0.35 + (msg_count/200)*0.3 + (emoji_rate*0.2) + (0.15 if has_history else 0)
        demo_prob = min(0.98, demo_prob)
        fig_gauge = go.Figure(go.Indicator(mode='gauge+number', value=demo_prob*100, title={'text':'Ghosting Probability'}, gauge={'axis':{'range':[0,100]}, 'steps':[{'range':[0,50],'color':'#b7f0c1'},{'range':[50,100],'color':'#ffd6d6'}], 'bar':{'color':'#3b1f6b'}}))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------ TAB: DATA & DOWNLOADS ------------------
with tabs[5]:
    st.subheader('📁 Data & Model Files')
    col1,col2 = st.columns(2)
    with col1:
        files = [f for f in os.listdir('.') if f.endswith(('.csv','.json','.pkl'))]
        for f in files:
            st.write(f)
            try: st.download_button(label=f'Download {f}', data=open(f,'rb'), file_name=f)
            except: pass
    with col2:
        uploaded = st.file_uploader('Upload JSON', type=['json'])
        if uploaded:
            try:
                data = json.load(uploaded)
                with open('model_performance.json','w') as out:
                    json.dump(data,out)
                st.success('Saved model_performance.json — reload app.')
            except Exception as e:
                st.error(f'Invalid JSON: {e}')

# ------------------ FOOTER ------------------
st.markdown('<div style="padding:8px 0; opacity:0.7; font-size:0.85rem">Made with ❤️ — Ghosting Research Dashboard</div>', unsafe_allow_html=True)
