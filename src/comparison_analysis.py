"""Compare TREE and synthetic Kaggle trait associations."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.formula.api as smf

from src.analysis_utils import regression_formula, report_dropna
from src.load_data import KAGGLE_TRAITS, OUTPUT_DIR, TREE_TRAITS, load_kaggle, load_tree

CONTROLS = ["age", "education_years", "parental_ses", "cognitive_score", "gpa", "job_performance", "sex"]


def _standardized_model(frame, outcome, traits, dataset_name):
    controls = [column for column in CONTROLS if column in frame.columns]
    missing = [column for column in CONTROLS if column not in frame.columns]
    if missing:
        print(f"{dataset_name}: controls unavailable and skipped for this dataset: {', '.join(missing)}")
    terms = list(traits) + [outcome] + controls
    clean = report_dropna(frame, terms, f"{dataset_name} standardized regression")
    continuous = list(traits) + [outcome] + [c for c in controls if c != "sex"]
    for column in continuous:
        sd = clean[column].std()
        if not sd or pd.isna(sd):
            raise ValueError(f"{dataset_name} cannot standardize constant or empty column {column!r}")
        clean[column] = (clean[column] - clean[column].mean()) / sd
    model = smf.ols(regression_formula(outcome, traits, controls), data=clean).fit(cov_type="HC3")
    rows = [{"dataset": dataset_name, "trait": trait, "std_beta": model.params[trait],
             "std_err": model.bse[trait], "p_value": model.pvalues[trait]} for trait in traits]
    return rows


def main():
    tree, kaggle = load_tree(), load_kaggle()
    results = _standardized_model(tree, "log_monthly_wage", list(TREE_TRAITS), "TREE")
    results += _standardized_model(kaggle, "log_income", KAGGLE_TRAITS, "Kaggle (synthetic)")
    betas = pd.DataFrame(results)
    betas["trait"] = betas["trait"].map({**TREE_TRAITS, **{t: t.title() for t in KAGGLE_TRAITS}})
    betas.to_csv(OUTPUT_DIR / "comparison_standardized_betas.csv", index=False)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=betas, x="trait", y="std_beta", hue="dataset", ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title="Standardized Big Five Coefficients by Dataset", xlabel="Trait", ylabel="Standardized beta")
    ax.tick_params(axis="x", rotation=15); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison_standardized_betas.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    correlations = []
    for dataset, frame, outcome, traits, labels in [
        ("TREE", tree, "log_monthly_wage", list(TREE_TRAITS), TREE_TRAITS),
        ("Kaggle (synthetic)", kaggle, "log_income", KAGGLE_TRAITS, {t: t.title() for t in KAGGLE_TRAITS}),
    ]:
        for trait in traits:
            clean = report_dropna(frame, [trait, outcome], f"{dataset} correlation: {trait}")
            r, p = pearsonr(clean[trait], clean[outcome])
            correlations.append({"dataset": dataset, "trait": labels[trait], "pearson_r": r, "p_value": p, "n": len(clean)})
    corr = pd.DataFrame(correlations)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=corr, x="trait", y="pearson_r", hue="dataset", ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title="Raw Big Five Correlations by Dataset", xlabel="Trait", ylabel="Pearson r")
    ax.tick_params(axis="x", rotation=15); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison_correlations.png", dpi=300, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
