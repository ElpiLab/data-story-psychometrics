"""Interactive descriptive page for the real TREE study."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.analysis_utils import report_dropna
from src.load_data import TREE_TRAITS, load_tree

st.set_page_config(page_title="TREE Study | Personality & Salary", layout="wide")
st.title("TREE Study: Personality and entry-level wages")


@st.cache_data
def get_tree():
    return load_tree()


df = get_tree()
wage_column = "monthly_wage" if "monthly_wage" in df.columns else "wage_hourly"
wage_data = report_dropna(df.assign(_display_wage=pd.to_numeric(df[wage_column], errors="coerce")), ["_display_wage"], "TREE wage histogram")
log_wage_data = report_dropna(df, ["log_monthly_wage"], "TREE log wage histogram")
left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(wage_data["_display_wage"], bins=40, ax=ax, color="#3B82F6")
    ax.set(title="Wage distribution", xlabel=wage_column, ylabel="Respondents")
    st.pyplot(fig); plt.close(fig)
with right:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(log_wage_data["log_monthly_wage"], bins=35, ax=ax, color="#10B981")
    ax.set(title="Log monthly wage distribution", xlabel="log_monthly_wage", ylabel="Respondents")
    st.pyplot(fig); plt.close(fig)

rows = []
for column, name in TREE_TRAITS.items():
    clean = report_dropna(df, [column, "log_monthly_wage"], f"TREE page correlation: {column}")
    rows.append({"Trait": name, "Correlation": clean[column].corr(clean["log_monthly_wage"])})
corr = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(corr["Trait"], corr["Correlation"], color=["#16A34A" if x > 0 else "#DC2626" for x in corr["Correlation"]])
ax.axhline(0, color="black", linewidth=1); ax.set_ylabel("Pearson r with log monthly wage")
st.pyplot(fig); plt.close(fig)
st.caption("Takeaway: direct personality–wage correlations are near zero at career entry.")
