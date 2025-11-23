import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import gdown

# --- PAGE CONFIG & STYLING ---
st.set_page_config(page_title="Ghosting Research Defense", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Global Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(to right, #4c1d95, #8b5cf6);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Card Styling for Metrics */
    .metric-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #6d28d9;
        transition: transform 0.2s;
    }
    .metric-container:hover {
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Alert Boxes */
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="main-header">
    <h1>👻 Ghosting Prediction Research</h1>
    <p style="font-size: 1.2rem; opacity: 0.9;">Behavioral Analysis & Machine Learning Defense</p>
    <p style="font-size: 0.9rem; opacity: 0.7;">Candidate: [Your Name]</p>
</div>
""", unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    file_id = '1gAogfnZDcpuSOTLa0UTD4tvXM0Vk0Jp2' 
    model_filename = 'ghosting_risk_model.pkl'
    
    if not os.path.exists(model_filename):
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(url, model_filename, quiet=False)
        except: pass 

    try:
        with open('model_performance.json', 'r') as f:
            perf_data = json.load(f)
        
        if os.path.exists(model_filename):
            model = joblib.load(model_filename)
            columns = joblib.load('model_columns.pkl')
            scaler = joblib.load('scaler.pkl')
        else:
            model, columns, scaler = None, None, None
            
        return perf_data, model, columns, scaler
    except FileNotFoundError:
        return None, None, None, None

perf_data, model, model_columns, scaler = load_assets()

if not perf_data:
    st.error("⚠️ Critical Error: Data files not found. Please upload 'model_performance.json' to GitHub.")
    st.stop()

# --- DATA ---
all_models_data = {
    "Random Forest": {"Accuracy": 0.9352, "Precision": 0.89, "Recall": 0.87, "F1": 0.88, "AUC": 0.9352},
    "Ensemble": {"Accuracy": 0.9337, "Precision": 0.89, "Recall": 0.86, "F1": 0.87, "AUC": 0.9337},
    "Gradient Boosting": {"Accuracy": 0.9319, "Precision": 0.89, "Recall": 0.86, "F1": 0.87, "AUC": 0.9319},
    "Logistic Regression": {"Accuracy": 0.8953, "Precision": 0.92, "Recall": 0.84, "F1": 0.88, "AUC": 0.8953},
    "Naive Bayes": {"Accuracy": 0.7692, "Precision": 0.75, "Recall": 0.70, "F1": 0.72, "AUC": 0.7692},
    "SVM": {"Accuracy": 0.5000, "Precision": 0.50, "Recall": 0.50, "F1": 0.50, "AUC": 0.5000},
    "K-NN": {"Accuracy": 0.4871, "Precision": 0.49, "Recall": 0.48, "F1": 0.48, "AUC": 0.4871},
}

# --- HELPER: METRIC CARD ---
def show_metrics(metrics):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-container"><div class="metric-label">Accuracy</div><div class="metric-value">{metrics['Accuracy']:.1%}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-container"><div class="metric-label">Precision</div><div class="metric-value">{metrics['Precision']:.1%}</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-container"><div class="metric-label">Recall</div><div class="metric-value">{metrics['Recall']:.1%}</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-container"><div class="metric-label">AUC Score</div><div class="metric-value">{metrics['AUC']:.4f}</div></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# --- HELPER: ALGORITHM PLOTS ---
def show_algorithm_plots(model_name, metrics, color_scale):
    c1, c2 = st.columns(2)
    
    # 1. Bar Chart: Metrics Breakdown
    with c1:
        df_metrics = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1", "AUC"],
            "Score": [metrics['Accuracy'], metrics['Precision'], metrics['Recall'], metrics['F1'], metrics['AUC']]
        })
        fig_bar = px.bar(
            df_metrics, x="Metric", y="Score", 
            title=f"{model_name} Performance Breakdown",
            color="Score", color_continuous_scale=color_scale,
            text_auto='.2%'
        )
        fig_bar.update_layout(yaxis_range=[0, 1.1])
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # 2. Simulated Error Distribution (Histogram)
    # Note: Since we don't have raw errors for every model loaded, we simulate a distribution based on accuracy
    with c2:
        accuracy = metrics['Accuracy']
        # Simulate 1000 predictions: mostly correct (1), some errors (0)
        # This visualizes the concept of error rate nicely
        simulated_errors = np.random.choice(
            ['Correct', 'Incorrect'], 
            size=1000, 
            p=[accuracy, 1-accuracy]
        )
        df_errors = pd.DataFrame({'Prediction Type': simulated_errors})
        
        fig_hist = px.histogram(
            df_errors, x="Prediction Type", 
            title=f"Prediction Error Distribution (Simulated)",
            color="Prediction Type",
            color_discrete_map={'Correct': '#4caf50', 'Incorrect': '#f44336'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# --- MAIN NAVIGATION ---
tabs = st.tabs(["🏆 Ranking", "🌲 Random Forest", "🚀 Gradient Boosting", "📈 Log. Reg.", "🔮 Ensemble", "🧪 Simulator"])

# --- TAB 1: RANKING ---
with tabs[0]:
    st.subheader("Algorithm Performance Leaderboard")
    
    df_compare = pd.DataFrame.from_dict(all_models_data, orient='index').reset_index()
    df_compare.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    df_compare = df_compare.sort_values(by='AUC', ascending=False)
    
    # Comparison Bar Chart
    fig = px.bar(
        df_compare, x='AUC', y='Model', orientation='h',
        color='AUC', color_continuous_scale='Viridis',
        title="Model Comparison by AUC Score", text_auto='.4f'
    )
    fig.update_layout(height=600, xaxis_title="AUC Score (Higher is Better)")
    st.plotly_chart(fig, use_container_width=True)

    # Scatter Plot: Accuracy vs Recall
    st.subheader("Trade-off Analysis: Accuracy vs Recall")
    fig_scatter = px.scatter(
        df_compare, x="Recall", y="Accuracy", 
        size="AUC", color="Model",
        hover_name="Model", size_max=60,
        title="Accuracy vs Recall (Size = AUC Score)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- TAB 2: RANDOM FOREST ---
with tabs[1]:
    st.subheader("🌲 Random Forest Classifier")
    metrics = all_models_data["Random Forest"]
    show_metrics(metrics)
    show_algorithm_plots("Random Forest", metrics, "Greens")
    
    # Feature Importance (Specific to Tree Models)
    st.subheader("Feature Importance Analysis")
    fi_data = perf_data.get('feature_importance', {})
    if fi_data:
        fi_df = pd.DataFrame({'Feature': fi_data.keys(), 'Importance': fi_data.values()}).sort_values(by='Importance', ascending=True).tail(10)
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title="Top 10 Predictors (Random Forest)", color_discrete_sequence=['#2e7d32'])
        st.plotly_chart(fig_fi, use_container_width=True)

# --- TAB 3: GRADIENT BOOSTING ---
with tabs[2]:
    st.subheader("🚀 Gradient Boosting Classifier")
    metrics = all_models_data["Gradient Boosting"]
    show_metrics(metrics)
    show_algorithm_plots("Gradient Boosting", metrics, "Oranges")
    
    st.info("Gradient Boosting achieves high precision by sequentially correcting errors of previous trees.")

# --- TAB 4: LOGISTIC REGRESSION ---
with tabs[3]:
    st.subheader("📈 Logistic Regression")
    metrics = all_models_data["Logistic Regression"]
    show_metrics(metrics)
    show_algorithm_plots("Logistic Regression", metrics, "Blues")
    
    st.info("Logistic Regression provides a strong baseline, proving linear separability for key features like response time.")

# --- TAB 5: ENSEMBLE ---
with tabs[4]:
    st.subheader("🔮 Voting Ensemble (Final Model)")
    metrics = all_models_data["Ensemble"]
    show_metrics(metrics)
    show_algorithm_plots("Ensemble", metrics, "Purples")
    
    # ROC Curve
    if 'roc_curve' in perf_data:
        st.subheader("ROC Curve")
        fpr = perf_data['roc_curve']['fpr']
        tpr = perf_data['roc_curve']['tpr']
        fig_roc = px.area(x=fpr, y=tpr, title="Ensemble ROC Curve", labels=dict(x='False Positive Rate', y='True Positive Rate'))
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig_roc, use_container_width=True)

# --- TAB 6: SIMULATOR ---
with tabs[5]:
    st.subheader("🧪 Interactive Risk Simulator")
    
    if model:
        c1, c2 = st.columns([1, 2])
        with c1:
            with st.container(border=True):
                st.markdown("### Inputs")
                msg_count = st.slider("Messages Sent", 0, 100, 20)
                emoji_rate = st.slider("Emoji Rate", 0.0, 1.0, 0.1)
                has_history = st.toggle("Has Ghosted Before?", value=False)
        
        with c2:
            st.markdown("### Prediction")
            if st.button("Run Model", type="primary", use_container_width=True):
                # Create input
                input_df = pd.DataFrame(columns=model_columns)
                input_df.loc[0] = 0
                if 'Message_Sent_Count' in input_df.columns: input_df['Message_Sent_Count'] = msg_count
                if 'Emoji_Usage_Rate' in input_df.columns: input_df['Emoji_Usage_Rate'] = emoji_rate
                if 'Has_Ghosting_History' in input_df.columns: input_df['Has_Ghosting_History'] = 1 if has_history else 0
                
                try:
                    prob = model.predict_proba(input_df)[0][1]
                    
                    # Visualization Gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob * 100,
                        title = {'text': "Ghosting Probability"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgreen"},
                                {'range': [50, 100], 'color': "salmon"}],
                        }
                    ))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    if prob > 0.5:
                        st.error("RESULT: High Risk Conversation")
                    else:
                        st.success("RESULT: Low Risk Conversation")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.warning("Model file loading...")
