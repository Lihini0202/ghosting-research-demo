import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import gdown

# ==========================================
# 1. PAGE CONFIGURATION & MODERN CSS
# ==========================================
st.set_page_config(
    page_title="GhostGuard AI Research",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="👻"
)

# Custom CSS for a "Dashboard" look
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Custom Card Container */
    .dashboard-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid #eef2f6;
    }
    
    /* Metrics Styling */
    .metric-label {
        font-size: 14px;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-delta {
        font-size: 14px;
        color: #10b981; /* Green */
        font-weight: 600;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4c1d95 !important;
        color: white !important;
    }
    
    /* Headings */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING
# ==========================================
@st.cache_resource
def load_assets():
    # 1. Download Model from Drive if missing
    file_id = '1gAogfnZDcpuSOTLa0UTD4tvXM0Vk0Jp2' # drive id
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
    st.error("⚠️ Data files missing. Please check your GitHub repository.")
    st.stop()

# --- Hardcoded Ranking Data (From your notebook) ---
all_models_data = {
    "Random Forest": 0.9352,
    "Ensemble": 0.9337,
    "Gradient Boosting": 0.9319,
    "Logistic Regression": 0.8953,
    "Naive Bayes": 0.7692,
    "SVM": 0.5000,
    "K-NN": 0.4871,
}

# ==========================================
# 3. SIDEBAR & HEADER
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712009.png", width=80)
    st.title("GhostGuard AI")
    st.markdown("---")
    st.markdown("**Project:** Ghosting Prediction")
    st.markdown("**Model:** Voting Ensemble")
    st.markdown("**Data Source:** 11,500 Interactions")
    st.markdown("---")
    st.success("System Status: Online 🟢")

st.markdown("# 📊 Behavioral Analysis Dashboard")
st.markdown("### Machine Learning Defense for Dating Apps")
st.markdown("---")

# ==========================================
# 4. MAIN TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["🏆 Performance Overview", "🧠 Feature Intelligence", "🧪 Live Simulator"])

# --- TAB 1: PERFORMANCE OVERVIEW ---
with tab1:
    # A. Top Metrics Row
    report = perf_data['classification_report']
    acc = report.get('accuracy', 0)
    weighted = report.get('weighted avg', {})
    
    # Use HTML columns for custom styling
    c1, c2, c3, c4 = st.columns(4)
    
    def custom_metric(label, value, subtext, col):
        col.markdown(f"""
        <div class="dashboard-card" style="text-align: center; padding: 15px;">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta">{subtext}</div>
        </div>
        """, unsafe_allow_html=True)

    custom_metric("Accuracy", f"{acc:.1%}", "vs Target 85% ↑", c1)
    custom_metric("Precision", f"{weighted.get('precision', 0):.1%}", "False Positive Rate ↓", c2)
    custom_metric("Recall", f"{weighted.get('recall', 0):.1%}", "Detection Rate ↑", c3)
    custom_metric("AUC Score", f"{all_models_data['Random Forest']:.4f}", "Best Model (RF)", c4)

    st.write("") # Spacing

    # B. Main Charts Row (Comparison + Confusion Matrix)
    col_left, col_right = st.columns([1.5, 1])
    
    with col_left:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### 🏎️ Algorithm Leaderboard")
        
        df_compare = pd.DataFrame(list(all_models_data.items()), columns=['Model', 'AUC'])
        df_compare = df_compare.sort_values(by='AUC', ascending=True)
        
        # Color the best model differently
        colors = ['#e0e7ff'] * (len(df_compare) - 1) + ['#4c1d95']
        
        fig_bar = go.Figure(go.Bar(
            x=df_compare['AUC'],
            y=df_compare['Model'],
            orientation='h',
            marker_color=colors,
            text=df_compare['AUC'].apply(lambda x: f"{x:.4f}"),
            textposition='auto'
        ))
        fig_bar.update_layout(
            plot_bgcolor='white',
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(range=[0.4, 1.0], title="AUC Score")
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Confusion Matrix")
        
        cm = np.array(perf_data.get('confusion_matrix', [[0,0],[0,0]]))
        
        # Custom Annotated Heatmap
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale=[[0, '#f3f4f6'], [1, '#4c1d95']],
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['No Ghost', 'Ghosted'],
            y=['No Ghost', 'Ghosted']
        )
        fig_cm.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("Darker purple = Higher volume of correct predictions.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: FEATURE INTELLIGENCE ---
with tab2:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("🧠 What Drives Ghosting?")
    st.write("Analysis of which behavioral factors contribute most to the risk score.")
    
    fi_data = perf_data.get('feature_importance', {})
    if fi_data:
        fi_df = pd.DataFrame({'Feature': fi_data.keys(), 'Importance': fi_data.values()})
        fi_df = fi_df.sort_values(by='Importance', ascending=True).tail(12)
        
        fig_fi = px.bar(
            fi_df, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Purples'
        )
        fig_fi.update_layout(height=600, plot_bgcolor='white')
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature Importance data is unavailable.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: LIVE SIMULATOR ---
with tab3:
    if model:
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("### 🎛️ Simulation Inputs")
            
            msg_count = st.slider("Messages Sent", 0, 100, 15)
            st.caption("Higher effort usually correlates with lower risk.")
            
            emoji_rate = st.slider("Emoji Usage Rate", 0.0, 1.0, 0.05)
            st.caption("Emotional expressiveness factor.")
            
            # Add inputs for other key features if they exist in your model
            # e.g., time_gap = st.number_input(...)
            
            st.markdown("---")
            has_history = st.toggle("Has Ghosting History?", value=False)
            
            if st.button("⚡ Calculate Risk", type="primary", use_container_width=True):
                # Run Prediction Logic
                input_df = pd.DataFrame(columns=model_columns)
                input_df.loc[0] = 0
                
                if 'Message_Sent_Count' in input_df.columns: input_df['Message_Sent_Count'] = msg_count
                if 'Emoji_Usage_Rate' in input_df.columns: input_df['Emoji_Usage_Rate'] = emoji_rate
                if 'Has_Ghosting_History' in input_df.columns: input_df['Has_Ghosting_History'] = 1 if has_history else 0
                
                try:
                    prob = model.predict_proba(input_df)[0][1]
                    st.session_state['prob'] = prob
                except Exception as e:
                    st.error(f"Error: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("### 🔮 AI Prediction")
            
            prob = st.session_state.get('prob', 0.5)
            
            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prob * 100,
                title = {'text': "Ghosting Risk Probability"},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#4c1d95"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': "#dcfce7"}, # Green
                        {'range': [50, 100], 'color': "#fee2e2"}], # Red
                }
            ))
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if prob > 0.5:
                st.error(f"⚠️ **High Risk ({prob:.1%})**: This user shows behavioral patterns strongly correlated with ghosting.")
            else:
                st.success(f"✅ **Low Risk ({prob:.1%})**: This interaction appears healthy and stable.")
                
            st.markdown('</div>', unsafe_allow_html=True)
            
    else:
        st.warning("⏳ Model is loading from Google Drive... please wait a moment.")
