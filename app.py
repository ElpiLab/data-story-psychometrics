"""Introduction page for the Streamlit data story."""
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Personality & Salary", layout="wide")
st.title("Personality & Salary")
intro_path = Path(__file__).resolve().parent / "Story Intro.txt"
if intro_path.exists():
    st.markdown(intro_path.read_text(encoding="utf-8"))
else:
    st.markdown("Explore how Big Five personality traits relate to wages and income across a real Swiss study and a synthetic comparison dataset.")
st.info("Use the sidebar to explore the TREE study, the synthetic Kaggle analysis, and the comparison.")
