# 👻 Ghosting Research Dashboard

**A Machine Learning & Behavioral Analysis Platform**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📌 Overview

This dashboard presents the findings of a research study on **predicting ghosting behavior** in dating app conversations. It serves as an interactive defense of the thesis, allowing users and supervisors to:

1.  **Visualize** model performance metrics (AUC, Precision, Recall).
2.  **Compare** 7 different algorithms (Random Forest, GBM, etc.).
3.  **Simulate** ghosting risk in real-time using a live prediction engine.
4.  **Analyze** behavioral feature importance.

The underlying model was trained on **11,500+ interactions** and achieves an accuracy of **~93.5%**.

---

## 🚀 Key Features

### 🏆 Algorithm Leaderboard
* A comparative analysis of multiple ML models.
* **Winner:** Random Forest / Ensemble (AUC 0.9352).
* **Visuals:** Bar charts ranking models by AUC, and scatter plots showing the Accuracy vs. Recall trade-off.

### 📊 Deep Dive Analytics
* **Confusion Matrix:** Heatmaps showing true positives vs false negatives.
* **ROC Curves:** Visualizing sensitivity vs specificity.
* **Feature Importance:** Identifying the top predictors (e.g., *Message Count*, *Response Time*, *History*).

### 🧪 Live Risk Simulator
* An interactive "Lab" where you can adjust sliders (e.g., number of messages, emoji usage).
* **Real-time Inference:** The app runs the pre-trained model on your inputs.
* **Output:** Returns a "Ghosting Probability" score with a visual gauge.

---

## 🛠️ Installation & Local Setup

To run this dashboard on your local machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/ghosting-research-dashboard.git](https://github.com/your-username/ghosting-research-dashboard.git)
cd ghosting-research-dashboard
```
Here is a professional README.md file tailored specifically for your Streamlit Research Dashboard.

You can copy this text, save it as README.md, and upload it to your GitHub repository along with your app files.

Markdown

# 👻 Ghosting Research Dashboard

**A Machine Learning & Behavioral Analysis Platform**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📌 Overview

This dashboard presents the findings of a research study on **predicting ghosting behavior** in dating app conversations. It serves as an interactive defense of the thesis, allowing users and supervisors to:

1.  **Visualize** model performance metrics (AUC, Precision, Recall).
2.  **Compare** 7 different algorithms (Random Forest, GBM, etc.).
3.  **Simulate** ghosting risk in real-time using a live prediction engine.
4.  **Analyze** behavioral feature importance.

The underlying model was trained on **11,500+ interactions** and achieves an accuracy of **~93.5%**.

---

## 🚀 Key Features

### 🏆 Algorithm Leaderboard
* A comparative analysis of multiple ML models.
* **Winner:** Random Forest / Ensemble (AUC 0.9352).
* **Visuals:** Bar charts ranking models by AUC, and scatter plots showing the Accuracy vs. Recall trade-off.

### 📊 Deep Dive Analytics
* **Confusion Matrix:** Heatmaps showing true positives vs false negatives.
* **ROC Curves:** Visualizing sensitivity vs specificity.
* **Feature Importance:** Identifying the top predictors (e.g., *Message Count*, *Response Time*, *History*).

### 🧪 Live Risk Simulator
* An interactive "Lab" where you can adjust sliders (e.g., number of messages, emoji usage).
* **Real-time Inference:** The app runs the pre-trained model on your inputs.
* **Output:** Returns a "Ghosting Probability" score with a visual gauge.

---

## 🛠️ Installation & Local Setup

To run this dashboard on your local machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/ghosting-research-dashboard.git](https://github.com/your-username/ghosting-research-dashboard.git)
cd ghosting-research-dashboard

2. Install Requirements
pip install -r requirements.txt

pip install -r requirements.txt
streamlit run app.py
```

☁️ Deployment (Streamlit Cloud)
This app is designed to be deployed on Streamlit Cloud for easy sharing.

Model Handling: Due to GitHub file size limits, the heavy Machine Learning model (ghosting_risk_model.pkl) is not hosted in this repo.

Auto-Download: When the app launches, it automatically downloads the model from a secure Google Drive link (configured in app.py).

📂 File Structure
app.py - The main Streamlit application code.

model_performance.json - Pre-computed metrics (Confusion Matrices, ROC data) exported from Jupyter.

model_columns.pkl - The exact feature names required by the model.

scaler.pkl - The standardization scaler used during training.

requirements.txt - List of Python dependencies.

📧 Dataset Access
This research utilizes a proprietary dataset of dating app conversations. The raw data is not public to protect user privacy.

For Researchers & Supervisors: If you wish to access the anonymized dataset for validation purposes, please contact the author directly.

Contact: [Edirisingha L] (lilysewmi@gmail.com)

📜 License
This project is for academic research purposes only.
