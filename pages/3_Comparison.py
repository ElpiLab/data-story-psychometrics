"""Compare estimates from TREE and the synthetic Kaggle dataset."""
import streamlit as st

from src.load_data import BASE_DIR

st.set_page_config(page_title="Comparison | Personality & Salary", layout="wide")
st.title("Comparing the two datasets")
for filename, caption in [
    ("comparison_standardized_betas.png", "Standardized multivariate coefficients"),
    ("comparison_correlations.png", "Raw trait and outcome correlations"),
]:
    image_path = BASE_DIR / "output" / filename
    if image_path.exists():
        st.image(str(image_path), caption=caption, width="stretch")
    else:
        st.info(f"{filename} is not available yet. Run `make analyze` to generate it.")
st.markdown("""
**Why the patterns differ:** TREE follows real Swiss participants around career entry, where fixed wage frameworks can leave little room for personality-linked pay differences. Its Big Five measures were collected at ages 15–16, so changes before employment can attenuate observed associations. The Kaggle comparison is synthetic and deliberately encodes stronger personality effects and control relationships; it illustrates a modeling contrast rather than an empirical estimate for a real population.
""")
