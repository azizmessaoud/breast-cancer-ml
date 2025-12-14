import streamlit as st
import pandas as pd
import numpy as np
from src.final_model import BreastCancerModel

st.set_page_config(page_title="🔬 Breast Cancer Detection", layout="wide")

st.title("🔬 Breast Cancer Detection")
st.info("97.66% Accuracy - Ensemble Voting Model")

# You can add more UI here...

