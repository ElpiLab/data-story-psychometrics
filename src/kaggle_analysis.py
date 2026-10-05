"""Correlations and controlled regression for the synthetic Kaggle dataset."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.formula.api as smf

from src.analysis_utils import report_dropna, regression_formula
from src.load_data import KAGGLE_TRAITS, OUTPUT_DIR, load_kaggle

CONTROLS = ["age", "education_years", "cognitive_score", "parental_ses", "job_performance", "sex"]


def main():
    df = load_kaggle()
    corr_data = report_dropna(df, KAGGLE_TRAITS + ["log_income"], "Kaggle correlations")
    correlations = []
    for trait in KAGGLE_TRAITS:
        r, p = pearsonr(corr_data[trait], corr_data["log_income"])
        correlations.append({"trait": trait, "pearson_r": r, "p_value": p, "n": len(corr_data)})
    pd.DataFrame(correlations).to_csv(OUTPUT_DIR / "kaggle_personality_log_income_corr.csv", index=False)

    controls = [column for column in CONTROLS if column in df.columns]
    missing = [column for column in CONTROLS if column not in df.columns]
    if missing:
        print(f"Kaggle controls unavailable and skipped: {', '.join(missing)}")
    model_data = report_dropna(df, KAGGLE_TRAITS + ["log_income"] + controls, "Kaggle multivariate regression")
    formula = regression_formula("log_income", KAGGLE_TRAITS, controls)
    model = smf.ols(formula, data=model_data).fit(cov_type="HC3")
    full_table = model.summary2().tables[1].reset_index(names="term")
    full_table.to_csv(OUTPUT_DIR / "kaggle_regression_summary.csv", index=False)
    coeffs = pd.DataFrame({"term": KAGGLE_TRAITS, "coefficient": [model.params[t] for t in KAGGLE_TRAITS],
                           "std_err": [model.bse[t] for t in KAGGLE_TRAITS], "p_value": [model.pvalues[t] for t in KAGGLE_TRAITS]})
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(coeffs["term"], coeffs["coefficient"], color=["#2563EB" if x >= 0 else "#DC2626" for x in coeffs["coefficient"]])
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title="Synthetic Kaggle: Big Five Coefficients", ylabel="Coefficient for log income", xlabel="Personality trait")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "kaggle_big5_multivariate_coefficients.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(model.summary())


if __name__ == "__main__":
    main()
