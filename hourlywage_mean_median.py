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
plt.xlabel("Log hourly wage")
plt.ylabel("Number of respondents")
plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_log_hourly_wage_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()