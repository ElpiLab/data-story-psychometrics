import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="The Paycheck Personality", layout="wide")

@st.cache_data
def load_tree():
    return pd.read_csv("TreeData/TREE_cleaned.csv")

tree = load_tree()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎮 The Game of Life",
    "🧠 What is Personality?",
    "🇨🇭 The Swiss Reality",
    "🔬 Controlled Analysis",
    "📊 The Kaggle Synthetic"
])

with tab1:
    st.title("The Paycheck Personality")
    st.markdown("""
    When you start a strategy game, you build your character from scratch.
    But in real life, you don't choose your traits. So does your personality
    determine your paycheck?
    """)

with tab2:
    st.header("What is Personality?")
    st.markdown("""
    **The Big Five (OCEAN):**
    - **Openness** — novelty, curiosity
    - **Conscientiousness** — discipline, organization
    - **Extraversion** — sociability, assertiveness
    - **Agreeableness** — cooperation, trust
    - **Neuroticism** — emotional reactivity
    """)

with tab3:
    st.header("The Swiss Reality: TREE Dataset")
    outcome = st.radio("Outcome:", ["Monthly Salary", "Hourly Wage"])
    y = "log_monthly" if outcome == "Monthly Salary" else "log_wage"
    
    traits = {
        "t0big5_o_comp": "Openness",
        "t0big5_c_comp": "Conscientiousness",
        "t0big5_e_comp": "Extraversion",
        "t0big5_a_comp": "Agreeableness",
        "t0big5_n_comp": "Neuroticism",
    }
    corrs = {name: tree[col].corr(tree[y]) for col, name in traits.items()}
    df = pd.DataFrame(list(corrs.items()), columns=["Trait", "Correlation"])
    fig = px.bar(df, x="Trait", y="Correlation", color="Correlation",
                 color_continuous_scale="RdBu")
    st.plotly_chart(fig, use_container_width=True)
    st.warning("Almost no direct correlation. Swiss entry-level salaries are standardized.")

with tab4:
    st.header("Controlled Analysis")
    st.markdown("Even after controlling for working hours, education, and job status, "
                "personality traits remain non-significant predictors of salary.")
    # load TREE2_cleaned.csv and show regression summary here if desired

with tab5:
    st.header("The Kaggle Synthetic Dataset")
    st.markdown("An illustrative model based on mainstream psychometric findings.")
    o = st.slider("Openness", 0, 100, 50)
    c = st.slider("Conscientiousness", 0, 100, 50)
    e = st.slider("Extraversion", 0, 100, 50)
    a = st.slider("Agreeableness", 0, 100, 50)
    n = st.slider("Neuroticism", 0, 100, 50)
    predicted = 50000 + c*200 + e*150 - a*100 - n*50 + o*50
    st.metric("Predicted Income (Synthetic)", f"${predicted:,.0f}")
    st.caption("⚠️ Synthetic, illustrative only.")

st.sidebar.info("Data Story Project · TREE (Uni Bern) + Kaggle Synthetic")