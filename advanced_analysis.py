import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_excel(DATA_DIR / "Complete_B5_Salary.xlsx")

# --------------------------------------------------
# 1. Audit columns and missingness
# --------------------------------------------------
print("Columns:")
print(df.columns.tolist())

print("\nMissing values (top 30):")
print(df.isna().sum().sort_values(ascending=False).head(30))

traits = {
    "t0big5_e_comp": "Extraversion",
    "t0big5_a_comp": "Agreeableness",
    "t0big5_c_comp": "Conscientiousness",
    "t0big5_n_comp": "Neuroticism",
    "t0big5_o_comp": "Openness",
}

# --------------------------------------------------
# 2. Build controls only if they exist in your data
# --------------------------------------------------
num_controls = [
    c for c in ["age", "age2", "tenure", "experience", "hours_worked", "work_hours"]
    if c in df.columns
]

cat_controls = [
    c for c in ["female", "gender", "sex", "education", "region",
                "occupation", "industry", "public_sector", "manager"]
    if c in df.columns
]

# Create age squared if age exists
if "age" in df.columns and "age2" not in df.columns:
    df["age2"] = df["age"] ** 2
    if "age2" not in num_controls:
        num_controls.append("age2")

# --------------------------------------------------
# 3. Complete-case regression dataset
# --------------------------------------------------
model_cols = list(traits.keys()) + ["log_wage"] + num_controls + cat_controls
model_cols = [c for c in model_cols if c in df.columns]

model_df = df.dropna(subset=model_cols).copy()
print(f"\nRegression observations: {len(model_df)}")

# --------------------------------------------------
# 4. OLS: log_wage ~ Big Five + controls
# --------------------------------------------------
terms = list(traits.keys()) + num_controls + [f"C({c})" for c in cat_controls]
formula = "log_wage ~ " + " + ".join(terms)

print("\nFormula:")
print(formula)

model = smf.ols(formula, data=model_df).fit(cov_type="HC3")
print(model.summary())

# --------------------------------------------------
# 5. Standardized Big Five coefficients
# --------------------------------------------------
sd_y = model_df["log_wage"].std()

std_rows = []
for t, name in traits.items():
    coef = model.params[t]
    ci = model.conf_int().loc[t]
    std_rows.append({
        "Trait": name,
        "Coef": coef,
        "Std_Coef": coef * model_df[t].std() / sd_y,
        "CI_low": ci[0],
        "CI_high": ci[1],
        "p_value": model.pvalues[t],
    })

std_df = pd.DataFrame(std_rows).sort_values("Std_Coef", ascending=False)
print("\nStandardized Big Five coefficients:")
print(std_df.to_string(index=False))

std_df.to_csv(
    OUTPUT_DIR / "04_big5_multivariate_coefficients.csv",
    index=False
)

# --------------------------------------------------
# 6. Coefficient plot
# --------------------------------------------------
plt.figure(figsize=(9, 5))
colors = ["#16A34A" if x > 0 else "#DC2626" for x in std_df["Std_Coef"]]
plt.barh(std_df["Trait"], std_df["Std_Coef"], color=colors)
plt.axvline(0, color="black", linewidth=1)
plt.xlabel("Standardized coefficient (SD change in log wage per 1 SD trait)")
plt.title("Big Five traits and log hourly wage, controlling for available covariates")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "04_big5_multivariate_coefficients.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# --------------------------------------------------
# 7. Correlation heatmap
# --------------------------------------------------
corr_cols = list(traits.keys()) + ["log_wage"]
corr_cols += [c for c in ["age", "female", "education"] if c in df.columns]
corr_cols = [c for c in corr_cols if c in df.columns]

corr = df[corr_cols].corr(numeric_only=True)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "04_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# --------------------------------------------------
# 8. Quantile regressions: 10th, 50th, 90th percentiles
# --------------------------------------------------
quant_terms = list(traits.keys()) + num_controls
quant_formula = "log_wage ~ " + " + ".join(quant_terms)

qmod = smf.quantreg(quant_formula, model_df)

for q in [0.10, 0.50, 0.90]:
    res = qmod.fit(q=q)
    print(f"\nQuantile {q:.2f}")
    print(res.params[list(traits.keys())])

# --------------------------------------------------
# 9. Optional: gender × trait interactions
# --------------------------------------------------
gender_col = next((c for c in ["female", "gender", "sex"] if c in model_df.columns), None)

if gender_col:
    terms_int = (
        list(traits.keys())
        + [f"{t}:C({gender_col})" for t in traits]
        + [f"C({gender_col})"]
        + num_controls
    )
    formula_int = "log_wage ~ " + " + ".join(terms_int)
    model_int = smf.ols(formula_int, data=model_df).fit(cov_type="HC3")
    print("\nInteraction model:")
    print(model_int.summary())