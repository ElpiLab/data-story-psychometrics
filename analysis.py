import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "dataset"

df = pd.read_excel(DATA_DIR / "Complete_B5_Salary.xlsx")

print(df.head())
print(df.shape)

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


# Hourly Wage Analysis
# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_excel(DATA_DIR / "Complete_B5_Salary.xlsx")

# --------------------------------------------------
# 2. Inspect wage variable
# --------------------------------------------------

print("\nHourly wage summary:")
print(df["wage_hourly"].describe())
print(f"\nMean hourly wage: CHF {df['wage_hourly'].mean():.2f}")
print(f"Median hourly wage: CHF {df['wage_hourly'].median():.2f}")
print(f"Maximum hourly wage: CHF {df['wage_hourly'].max():.2f}")

print("\nHourly wage percentiles:")
print(
    df["wage_hourly"].quantile(
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )
)

# --------------------------------------------------
# 3. Create chart
# --------------------------------------------------

sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="wage_hourly",
    bins=50,
    color="#3B82F6",
    edgecolor="white"
)

plt.axvline(
    df["wage_hourly"].median(),
    color="#EF4444",
    linestyle="--",
    linewidth=2,
    label=f"Median: CHF {df['wage_hourly'].median():.2f}"
)

plt.title("Distribution of Hourly Wages")
plt.xlabel("Hourly wage in CHF")
plt.ylabel("Number of respondents")
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_hourly_wage_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# --------------------------------------------------
# Insight 2: Distribution of log hourly wage
# --------------------------------------------------

print("\nLog hourly wage summary:")
print(df["log_wage"].describe())

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="log_wage",
    bins=35,
    color="#10B981",
    edgecolor="white",
    kde=True
)

plt.axvline(
    df["log_wage"].median(),
    color="#EF4444",
    linestyle="--",
    linewidth=2,
    label=f"Median log wage: {df['log_wage'].median():.2f}"
)

plt.title("Distribution of Log Hourly Wages")
plt.xlabel("Logarithm hourly wage, ln(CHF per hour)")
plt.ylabel("Number of respondents")
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_log_hourly_wage_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# --------------------------------------------------
# Insight 3: Correlation between personality and wage
# --------------------------------------------------

traits = {
    "t0big5_e_comp": "Extraversion",
    "t0big5_a_comp": "Agreeableness",
    "t0big5_c_comp": "Conscientiousness",
    "t0big5_n_comp": "Neuroticism",
    "t0big5_o_comp": "Openness",
}

# Keep only rows where all five personality scores
# and log hourly wage are available
analysis_df = df.dropna(
    subset=list(traits.keys()) + ["log_wage"]
).copy()

print("\n" + "=" * 50)
print("INSIGHT 3: PERSONALITY AND LOG HOURLY WAGE")
print("=" * 50)

print(f"\nObservations used: {len(analysis_df)}")

correlation_results = []

for column, trait_name in traits.items():
    correlation = analysis_df[column].corr(analysis_df["log_wage"])

    correlation_results.append(
        {
            "Trait": trait_name,
            "Correlation with log wage": correlation
        }
    )

correlation_df = pd.DataFrame(correlation_results)

correlation_df = correlation_df.sort_values(
    by="Correlation with log wage",
    ascending=False
)

print("\nCorrelation results:")
print(correlation_df.to_string(index=False))

# Save the correlation table
correlation_df.to_csv(
    OUTPUT_DIR / "03_personality_wage_correlations.csv",
    index=False
)

# Create chart
plt.figure(figsize=(10, 6))

bar_colors = [
    "#16A34A" if value > 0 else "#DC2626"
    for value in correlation_df["Correlation with log wage"]
]

plt.bar(
    correlation_df["Trait"],
    correlation_df["Correlation with log wage"],
    color=bar_colors
)

plt.axhline(
    y=0,
    color="black",
    linewidth=1
)

plt.title("Big Five Personality Traits and Log Hourly Wage")
plt.xlabel("Personality trait")
plt.ylabel("Pearson correlation with log hourly wage")
plt.xticks(rotation=20, ha="right")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_personality_wage_correlations.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()