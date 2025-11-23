import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.express as px
import gdown
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Ghosting Research Defense", layout="wide")

st.title("👻 Dating App Ghosting Prediction Research")
st.markdown("**Candidate:** [Your Name] | **Topic:** Behavioral Analysis of Ghosting")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # 1. Define Google Drive File ID for the Model
    # REPLACE THIS WITH YOUR ACTUAL FILE ID FROM STEP 1
    file_id = '1gAogfnZDcpuSOTLa0UTD4tvXM0Vk0Jp2' 
    model_filename = 'ghosting_risk_model.pkl'
    
    # 2. Download Model if missing
    if not os.path.exists(model_filename):
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(url, model_filename, quiet=False)
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None, None, None, None

    try:
        # 3. Load Metrics
        with open('model_performance.json', 'r') as f:
            perf_data = json.load(f)
        
        # 4. Load Model & Scalers
        model = joblib.load(model_filename)
        columns = joblib.load('model_columns.pkl')
        scaler = joblib.load('scaler.pkl')
        
        return perf_data, model, columns, scaler
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
        return None, None, None, None

# Load everything
perf_data, model, model_columns, scaler = load_assets()

if not perf_data:
    st.stop()

# --- NAVIGATION ---
tabs = st.tabs(["📊 Model Performance (Plots)", "🧪 Live Simulation"])

# =========================================
# TAB 1: PERFORMANCE PLOTS
# =========================================
with tabs[0]:
    st.header("Model Evaluation")
    
    # 1. Metrics Row
    report = perf_data['classification_report']
    # Handle cases where 'accuracy' is a direct key or nested
    acc = report.get('accuracy', 0)
    weighted = report.get('weighted avg', {})
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Accuracy", f"{acc:.1%}")
    col2.metric("Precision", f"{weighted.get('precision', 0):.1%}")
    col3.metric("Recall", f"{weighted.get('recall', 0):.1%}")
    col4.metric("F1 Score", f"{weighted.get('f1-score', 0):.1%}")
    
    st.divider()

    col_left, col_right = st.columns(2)

    # 2. Feature Importance Chart
    with col_left:
        st.subheader("What causes ghosting?")
        fi_data = perf_data.get('feature_importance', {})
        if fi_data:
            fi_df = pd.DataFrame({
                'Feature': fi_data.keys(),
                'Importance': fi_data.values()
            }).sort_values(by='Importance', ascending=True).tail(15)

            fig_fi = px.bar(
                fi_df, x='Importance', y='Feature', orientation='h',
                title="Top 15 Predictive Features", color='Importance'
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.warning("Feature Importance data missing.")

    # 3. Confusion Matrix
    with col_right:
        st.subheader("Prediction Accuracy Matrix")
        cm = np.array(perf_data['confusion_matrix'])
        fig_cm = px.imshow(
            cm, text_auto=True, aspect="auto",
            x=['Predicted No-Ghost', 'Predicted Ghost'],
            y=['Actual No-Ghost', 'Actual Ghost'],
            color_continuous_scale='Blues',
            title="Confusion Matrix"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # 4. ROC Curve
    if 'roc_curve' in perf_data:
        st.subheader("ROC Curve Analysis")
        fpr = perf_data['roc_curve']['fpr']
        tpr = perf_data['roc_curve']['tpr']
        fig_roc = px.area(x=fpr, y=tpr, title="ROC Curve", labels=dict(x='False Positive Rate', y='True Positive Rate'))
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig_roc, use_container_width=True)

# =========================================
# TAB 2: LIVE SIMULATOR
# =========================================
with tabs[1]:
    st.header("Test the Model")
    st.write("Adjust the parameters below to see how the model predicts risk in real-time.")

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("User Behavior")
        msg_count = st.slider("Messages Sent", 0, 100, 10)
        emoji_rate = st.slider("Emoji Usage Rate", 0.0, 1.0, 0.1)
        # Add other inputs matching your model_columns if necessary
        
        # Simulate history input if it's a feature
        has_history = st.checkbox("Has Ghosted Before?", value=False)

    with col_result:
        st.subheader("Prediction")
        
        if st.button("Run Prediction"):
            # Create Input Data
            input_df = pd.DataFrame(columns=model_columns)
            input_df.loc[0] = 0 # Initialize with 0
            
            # Map Inputs - Ensure these column names match your notebook EXACTLY
            if 'Message_Sent_Count' in input_df.columns:
                input_df['Message_Sent_Count'] = msg_count
            if 'Emoji_Usage_Rate' in input_df.columns:
                input_df['Emoji_Usage_Rate'] = emoji_rate
            # Add logic for other features used in training
            if 'Has_Ghosting_History' in input_df.columns:
                input_df['Has_Ghosting_History'] = 1 if has_history else 0

            # Predict
            try:
                prob = model.predict_proba(input_df)[0][1] 
                
                st.metric("Ghosting Probability", f"{prob:.1%}")
                
                if prob > 0.5:
                    st.error(f"⚠️ HIGH RISK DETECTED\nThe model predicts a {prob:.1%} chance of ghosting.")
                else:
                    st.success(f"✅ LOW RISK\nOnly a {prob:.1%} chance of ghosting.")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
