import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import gdown

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION (Wide Mode & Modern Title)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Ghosting Prediction Research",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="👻"
)

# ---------------------------------------------------------
# 2. CUSTOM CSS (The "Modern Data Lab" Look)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }
    
    /* Background */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Header Styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #4c1d95, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #64748b;
        margin-bottom: 2rem;
    }

    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1e293b;
    }
    
    /* Chart Containers */
    .chart-container {
        background: white;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #f1f5f9;
        margin-bottom: 30px;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        font-weight: 600;
        font-size: 1rem;
        border: 1px solid #e2e8f0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4c1d95;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. LOAD ASSETS
# ---------------------------------------------------------
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
    st.error("⚠️ Data Missing. Please upload 'model_performance.json'.")
    st.stop()

# --- HARDCODED MODEL DATA ---
all_models_data = {
    "Random Forest": {"Accuracy": 0.9352, "Precision": 0.89, "Recall": 0.87, "F1": 0.88, "AUC": 0.9352},
    "Ensemble": {"Accuracy": 0.9337, "Precision": 0.89, "Recall": 0.86, "F1": 0.87, "AUC": 0.9337},
    "Gradient Boosting": {"Accuracy": 0.9319, "Precision": 0.89, "Recall": 0.86, "F1": 0.87, "AUC": 0.9319},
    "Logistic Regression": {"Accuracy": 0.8953, "Precision": 0.92, "Recall": 0.84, "F1": 0.88, "AUC": 0.8953},
    "Naive Bayes": {"Accuracy": 0.7692, "Precision": 0.75, "Recall": 0.70, "F1": 0.72, "AUC": 0.7692},
    "SVM": {"Accuracy": 0.5000, "Precision": 0.50, "Recall": 0.50, "F1": 0.50, "AUC": 0.5000},
    "K-NN": {"Accuracy": 0.4871, "Precision": 0.49, "Recall": 0.48, "F1": 0.48, "AUC": 0.4871},
}

# ---------------------------------------------------------
# 4. VISUALIZATION HELPERS (BIGGER & BOLDER)
# ---------------------------------------------------------
def make_modern_chart(fig, height=600):
    fig.update_layout(
        height=height,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family="Inter", size=14, color="#334155"),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=True, gridcolor='#f1f5f9', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9', zeroline=False),
        hoverlabel=dict(bgcolor="white", font_size=14, font_family="Inter")
    )
    return fig

def metric_card_row(metrics):
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Accuracy", metrics['Accuracy'], c1),
        ("Precision", metrics['Precision'], c2),
        ("Recall", metrics['Recall'], c3),
        ("AUC Score", metrics['AUC'], c4)
    ]
    for label, val, col in cards:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. APP LAYOUT
# ---------------------------------------------------------
st.markdown('<div class="main-title">Ghosting Research Defense</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Machine Learning Analysis of Dating Behavior</div>', unsafe_allow_html=True)

tabs = st.tabs(["🏆 Ranking", "🌲 Random Forest", "🚀 Gradient Boosting", "📈 Logistic Reg.", "🔮 Ensemble", "🧪 Simulator"])

# --- TAB 1: RANKING ---
with tabs[0]:
    st.markdown("### 🥇 Algorithm Performance Comparison")
    
    df_compare = pd.DataFrame.from_dict(all_models_data, orient='index').reset_index()
    df_compare.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    df_compare = df_compare.sort_values(by='AUC', ascending=True)

    # Big Horizontal Bar Chart
    with st.container():
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_compare = px.bar(
            df_compare, y='Model', x='AUC', orientation='h',
            color='AUC', color_continuous_scale='Viridis',
            text_auto='.4f', title="<b>AUC Score Leaderboard</b> (Test Set)"
        )
        fig_compare.update_traces(textfont_size=16, textposition='outside')
        fig_compare.update_layout(xaxis_range=[0.4, 1.0])
        st.plotly_chart(make_modern_chart(fig_compare, height=650), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Comparison Scatter
    with st.container():
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            df_compare, x="Recall", y="Precision", size="Accuracy",
            color="Model", size_max=50,
            title="<b>Precision vs Recall Trade-off</b> (Bubble Size = Accuracy)",
            hover_name="Model"
        )
        st.plotly_chart(make_modern_chart(fig_scatter, height=500), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- GENERIC TAB BUILDER FUNCTION ---
def build_model_tab(model_name, color_theme):
    metrics = all_models_data[model_name]
    
    # 1. Metrics Cards
    metric_card_row(metrics)
    
    col_left, col_right = st.columns(2)
    
    # 2. Performance Breakdown (Bar)
    with col_left:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        df_metrics = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1", "AUC"],
            "Score": [metrics['Accuracy'], metrics['Precision'], metrics['Recall'], metrics['F1'], metrics['AUC']]
        })
        fig_bar = px.bar(
            df_metrics, x="Metric", y="Score", 
            title=f"<b>{model_name} Performance Breakdown</b>",
            color="Score", color_continuous_scale=color_theme,
            text_auto='.2%'
        )
        fig_bar.update_layout(yaxis_range=[0, 1.1])
        st.plotly_chart(make_modern_chart(fig_bar, height=500), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Confusion Matrix (Heatmap)
    with col_right:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        cm = np.array(perf_data.get('confusion_matrix', [[0,0],[0,0]]))
        
        fig_cm = px.imshow(
            cm, text_auto=True, 
            color_continuous_scale=color_theme,
            title=f"<b>Confusion Matrix ({model_name})</b>",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['No Ghost', 'Ghosted'], y=['No Ghost', 'Ghosted']
        )
        fig_cm.update_traces(textfont_size=20)
        st.plotly_chart(make_modern_chart(fig_cm, height=500), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 4. Feature Importance (Full Width)
    if 'feature_importance' in perf_data:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fi_data = perf_data['feature_importance']
        fi_df = pd.DataFrame({'Feature': fi_data.keys(), 'Importance': fi_data.values()})
        fi_df = fi_df.sort_values(by='Importance', ascending=True).tail(15)

        fig_fi = px.bar(
            fi_df, x='Importance', y='Feature', orientation='h',
            title=f"<b>Top 15 Predictive Features ({model_name})</b>",
            color='Importance', color_continuous_scale=color_theme
        )
        st.plotly_chart(make_modern_chart(fig_fi, height=700), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- BUILD INDIVIDUAL TABS ---
with tabs[1]: build_model_tab("Random Forest", "Greens")
with tabs[2]: build_model_tab("Gradient Boosting", "Oranges")
with tabs[3]: build_model_tab("Logistic Regression", "Blues")
with tabs[4]: build_model_tab("Ensemble", "Purples")

# --- TAB 6: SIMULATOR ---
with tabs[5]:
    if model:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 🧪 Live Prediction Lab")
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            msg_count = st.slider("Messages Sent", 0, 100, 20)
            emoji_rate = st.slider("Emoji Rate", 0.0, 1.0, 0.1)
            time_gap = st.number_input("Response Time (hours)", 0.0, 48.0, 2.0)
            has_history = st.toggle("Has Ghosted Before?", False)
            
            if st.button("Run Analysis", type="primary", use_container_width=True):
                # Input mapping
                input_df = pd.DataFrame(columns=model_columns)
                input_df.loc[0] = 0
                if 'Message_Sent_Count' in input_df.columns: input_df['Message_Sent_Count'] = msg_count
                if 'Emoji_Usage_Rate' in input_df.columns: input_df['Emoji_Usage_Rate'] = emoji_rate
                if 'Has_Ghosting_History' in input_df.columns: input_df['Has_Ghosting_History'] = 1 if has_history else 0
                
                try:
                    prob = model.predict_proba(input_df)[0][1]
                    st.session_state['sim_prob'] = prob
                except:
                    st.session_state['sim_prob'] = 0.5
        
        with c2:
            prob = st.session_state.get('sim_prob', 0.0)
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': "Risk Probability"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#4c1d95"},
                    'steps': [
                        {'range': [0, 50], 'color': "#dcfce7"},
                        {'range': [50, 100], 'color': "#fee2e2"}],
                }
            ))
            st.plotly_chart(make_modern_chart(fig_gauge, height=400), use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Model Loading...")
