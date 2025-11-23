import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Ghosting Research Defense", layout="wide")

st.title("👻 Dating App Ghosting Prediction Research")
st.markdown("**Candidate:** [Your Name] | **Topic:** Behavioral Analysis of Ghosting")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        # Load Metrics
        with open('model_performance.json', 'r') as f:
            perf_data = json.load(f)
        
        # Load Model
        model = joblib.load('ghosting_risk_model.pkl')
        columns = joblib.load('model_columns.pkl')
        scaler = joblib.load('scaler.pkl')
        
        return perf_data, model, columns, scaler
    except FileNotFoundError:
        return None, None, None, None

perf_data, model, model_columns, scaler = load_assets()

if not perf_data:
    st.error("❌ Critical Error: Missing files. Please ensure .json and .pkl files are in the GitHub repo.")
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
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Accuracy", f"{report['accuracy']:.1%}")
    col2.metric("Precision", f"{report['weighted avg']['precision']:.1%}")
    col3.metric("Recall", f"{report['weighted avg']['recall']:.1%}")
    col4.metric("F1 Score", f"{report['weighted avg']['f1-score']:.1%}")
    
    st.divider()

    col_left, col_right = st.columns(2)

    # 2. Feature Importance Chart
    with col_left:
        st.subheader("What causes ghosting?")
        fi_data = perf_data['feature_importance']
        fi_df = pd.DataFrame({
            'Feature': fi_data.keys(),
            'Importance': fi_data.values()
        }).sort_values(by='Importance', ascending=True).tail(15)

        fig_fi = px.bar(
            fi_df, x='Importance', y='Feature', orientation='h',
            title="Top 15 Predictive Features", color='Importance'
        )
        st.plotly_chart(fig_fi, use_container_width=True)

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
    st.subheader("ROC Curve Analysis")
    fpr = perf_data['roc_curve']['fpr']
    tpr = perf_data['roc_curve']['tpr']
    fig_roc = px.area(x=fpr, y=tpr, title="ROC Curve (Sensitivity vs 1-Specificity)", labels=dict(x='False Positive Rate', y='True Positive Rate'))
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
        response_time = st.number_input("Avg Response Time (minutes)", 0, 5000, 60)
        
        # Simulate history input
        has_history = st.checkbox("Has Ghosted Before?", value=False)

    with col_result:
        st.subheader("Prediction")
        
        # Create Input Data
        input_df = pd.DataFrame(columns=model_columns)
        input_df.loc[0] = 0 # Initialize with 0
        
        # Map Inputs
        if 'Message_Sent_Count' in input_df.columns:
            input_df['Message_Sent_Count'] = msg_count
        if 'Emoji_Usage_Rate' in input_df.columns:
            input_df['Emoji_Usage_Rate'] = emoji_rate
        if 'Avg_Response_Time' in input_df.columns:
            input_df['Avg_Response_Time'] = response_time
        if 'Has_Ghosting_History' in input_df.columns:
            input_df['Has_Ghosting_History'] = 1 if has_history else 0

        # Scale (if your model used scaling)
        # input_scaled = scaler.transform(input_df) 
        
        # Predict
        if st.button("Run Prediction"):
            # Note: Using input_df directly. If you scaled in notebook, use input_scaled
            prob = model.predict_proba(input_df)[0][1] 
            
            st.metric("Ghosting Probability", f"{prob:.1%}")
            
            if prob > 0.5:
                st.error(f"⚠️ HIGH RISK DETECTED\nThe model predicts a {prob:.1%} chance of ghosting.")
            else:
                st.success(f"✅ LOW RISK\nOnly a {prob:.1%} chance of ghosting.")
