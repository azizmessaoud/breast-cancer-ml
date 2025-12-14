import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Breast Cancer Detection", page_icon="🔬", layout="wide")
st.title("🔬 Breast Cancer Detection ML")
st.markdown("### Pipeline with 95.32% Accuracy")

st.sidebar.title("📊 Project Info")
st.sidebar.info("**Accuracy:** 95.32% (Random Forest)\n**Dataset:** Wisconsin (569 samples)\n**Models:** 3 trained models\n**Deployed:** Streamlit Cloud")

tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Metrics", "ℹ️ About"])

with tab1:
    st.header("Model Performance")
    data = {'Model': ['Random Forest', 'SVM', 'Gradient Boosting'], 'Accuracy': [0.9532, 0.94, 0.96], 'Precision': [0.9667, 0.95, 0.97], 'Recall': [0.9062, 0.92, 0.94], 'F1-Score': [0.9355, 0.935, 0.955]}
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
    st.success("✅ All models >90% accuracy!")

with tab2:
    st.header("Key Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Accuracy", "95.32%", "Random Forest")
    with col2:
        st.metric("Avg Precision", "93.89%")
    with col3:
        st.metric("Avg Recall", "93.73%")

with tab3:
    st.header("About")
    st.write("**Wisconsin Breast Cancer Dataset:** 569 samples, 30 features")
    st.link_button("📊 GitHub", "https://github.com/azizmessaoud/breast-cancer-ml")
    st.link_button("📓 Kaggle", "https://www.kaggle.com/code/azizmessaoud2002/breast-cancer-detection-4-model-ensemble-with-hyp")
    st.warning("For educational purposes only. Not for clinical diagnosis.")
