"""TREE descriptive and controlled wage analyses."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.analysis_utils import fit_tree, report_dropna
from src.load_data import OUTPUT_DIR, TREE_TRAITS, load_tree


def main():
    df = load_tree()
    wage_col = "monthly_wage" if "monthly_wage" in df.columns else "wage_hourly"
    if wage_col not in df.columns:
        raise ValueError("TREE data needs monthly_wage (preferred) or wage_hourly for distributions")
    wage = pd.to_numeric(df[wage_col], errors="coerce")
    print(f"TREE wage distribution: n before dropna={len(df)}; n after dropna={wage.notna().sum()}")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(wage.dropna(), bins=40, color="#3B82F6", ax=ax)
    ax.set(title=f"TREE {wage_col.replace('_', ' ').title()} Distribution", xlabel=wage_col, ylabel="Respondents")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_hourly_wage_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    log_wage = df["log_monthly_wage"]
    print(f"TREE log wage distribution: n before dropna={len(df)}; n after dropna={log_wage.notna().sum()}")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(log_wage.dropna(), bins=35, color="#10B981", ax=ax)
    ax.set(title="TREE Log Monthly Wage Distribution", xlabel="Log monthly wage", ylabel="Respondents")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_log_hourly_wage_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    analysis = report_dropna(df, list(TREE_TRAITS) + ["log_monthly_wage"], "TREE correlations")
    rows = [{"Trait": name, "Correlation with log wage": analysis[column].corr(analysis["log_monthly_wage"])} for column, name in TREE_TRAITS.items()]
    corr = pd.DataFrame(rows).sort_values("Correlation with log wage")
    corr.to_csv(OUTPUT_DIR / "03_personality_wage_correlations.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(corr["Trait"], corr["Correlation with log wage"], color=["#16A34A" if x > 0 else "#DC2626" for x in corr["Correlation with log wage"]])
    ax.axhline(0, color="black", linewidth=1)
    ax.set(title="TREE Personality and Log Monthly Wage", ylabel="Pearson correlation", xlabel="Trait")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_personality_wage_correlations.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    model, clean = fit_tree(df)
    sd_y = clean["log_monthly_wage"].std()
    results = []
    for column, name in TREE_TRAITS.items():
        interval = model.conf_int().loc[column]
        results.append({"Trait": name, "Coef": model.params[column], "Std_Coef": model.params[column] * clean[column].std() / sd_y,
                        "CI_low": interval.iloc[0], "CI_high": interval.iloc[1], "p_value": model.pvalues[column]})
    table = pd.DataFrame(results).sort_values("Std_Coef", ascending=False)
    table.to_csv(OUTPUT_DIR / "04_big5_multivariate_coefficients.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(table["Trait"], table["Std_Coef"], color=["#16A34A" if x > 0 else "#DC2626" for x in table["Std_Coef"]])
    ax.axvline(0, color="black", linewidth=1)
    ax.invert_yaxis()
    ax.set(title="TREE Big Five and Log Monthly Wage", xlabel="Standardized coefficient (robust OLS)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_big5_multivariate_coefficients.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    numeric = [*TREE_TRAITS, "log_monthly_wage"]
    heat = report_dropna(df, numeric, "TREE correlation heatmap")[numeric].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(model.summary())


if __name__ == "__main__":
    main()
