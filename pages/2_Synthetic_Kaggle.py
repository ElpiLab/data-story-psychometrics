"""Interactive controlled model for the synthetic comparison data."""
import plotly.express as px
import streamlit as st

from src.analysis_utils import fit_kaggle
from src.load_data import KAGGLE_TRAITS, load_kaggle

st.set_page_config(page_title="Synthetic Kaggle | Personality & Salary", layout="wide")
st.title("Synthetic Kaggle: what changes with controls?")


@st.cache_data
def get_kaggle():
    return load_kaggle()


@st.cache_data
def cached_fit(controls):
    return fit_kaggle(get_kaggle(), controls)


data = get_kaggle()
control_options = ["age", "education_years", "parental_ses", "cognitive_score", "gpa", "job_performance", "sex"]
control_options = [column for column in control_options if column in data.columns]
controls = st.multiselect("Controls", control_options, default=[x for x in ["age", "education_years"] if x in control_options])
model, clean = cached_fit(tuple(controls))
coefficients = [{"term": trait, "coefficient": model.params[trait]} for trait in KAGGLE_TRAITS]
fig = px.bar(coefficients, x="term", y="coefficient", color="coefficient", color_continuous_scale="RdBu",
             title="Big Five coefficients for log income")
fig.add_hline(y=0, line_color="black")
st.plotly_chart(fig, width="stretch")
table = model.summary2().tables[1].reset_index(names="term")
st.dataframe(table, width="stretch", hide_index=True)
st.caption("This Kaggle dataset is synthetic; its patterns illustrate how added controls can change estimated associations.")
