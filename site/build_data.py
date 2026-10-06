"""Prepare data and a local preview for the Quarto/GitHub Pages story."""
import json
import html
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.formula.api as smf

from src.analysis_utils import regression_formula, report_dropna
from src.load_data import KAGGLE_TRAITS, TREE_TRAITS, load_kaggle, load_tree

SITE_DIR = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
CONTROLS = ["age", "education_years", "parental_ses", "cognitive_score", "gpa", "job_performance", "sex"]


def render_teammate_text():
    """Render the teammate's plain-text article without changing its wording."""
    raw = (ROOT / "Story Intro.txt").read_text(encoding="utf-8")
    heading_lines = {"Story Intro", "What Is Personality?", "Kaggle Dataset"}
    blocks = [block.strip() for block in raw.split("\n\n") if block.strip()]
    rendered = []
    for block in blocks:
        content = html.escape(block).replace("\n", "<br>\n")
        if block in heading_lines:
            rendered.append(f"<h2>{content}</h2>")
        else:
            rendered.append(f"<p>{content}</p>")
    return "\n".join(rendered)


def build_story_data():
    tree = load_tree()
    tree_complete = report_dropna(tree, [*TREE_TRAITS, "log_monthly_wage"], "TREE headline sample")
    correlations = []
    for column, label in TREE_TRAITS.items():
        pair = report_dropna(tree, [column, "log_monthly_wage"], f"TREE correlation: {label}")
        r, p = pearsonr(pair[column], pair["log_monthly_wage"])
        # Fisher's z approximation gives a readable uncertainty interval for each r.
        half_width = 1.96 / np.sqrt(max(len(pair) - 3, 1))
        correlations.append({"trait": label, "r": float(r), "p": float(p), "n": len(pair), "ci": float(half_width)})
    monthly = pd.to_numeric(tree_complete["monthly_wage"], errors="coerce")
    monthly_clean = report_dropna(tree_complete.assign(_monthly=monthly), ["_monthly"], "TREE analysis-sample monthly wage summary")

    kaggle = load_kaggle()
    model_specs = {
        "traits": {"label": "Traits only", "controls": []},
        "demographics": {"label": "+ Age and education", "controls": ["age", "education_years"]},
        "full": {"label": "+ Full control set", "controls": CONTROLS},
    }
    models = {}
    for key, spec in model_specs.items():
        controls = [column for column in spec["controls"] if column in kaggle.columns]
        missing = [column for column in spec["controls"] if column not in kaggle.columns]
        if missing:
            print(f"Kaggle model {key}: missing controls skipped: {', '.join(missing)}")
        cols = KAGGLE_TRAITS + ["log_income"] + controls
        clean = report_dropna(kaggle, cols, f"Kaggle model {key}")
        standardized = clean.copy()
        continuous = KAGGLE_TRAITS + ["log_income"] + [col for col in controls if col != "sex"]
        for column in continuous:
            spread = standardized[column].std()
            if not spread or pd.isna(spread):
                raise ValueError(f"Cannot standardize constant column {column!r} in model {key}")
            standardized[column] = (standardized[column] - standardized[column].mean()) / spread
        formula = regression_formula("log_income", KAGGLE_TRAITS, controls)
        fitted = smf.ols(formula, data=standardized).fit(cov_type="HC3")
        models[key] = {
            "label": spec["label"],
            "controls_label": ", ".join(controls) if controls else "no controls",
            "n": int(fitted.nobs),
            "rows": [{"trait": trait.title(), "beta": float(fitted.params[trait]), "se": float(fitted.bse[trait]),
                      "p": float(fitted.pvalues[trait])} for trait in KAGGLE_TRAITS],
        }
    return {"tree": {"n": int(tree_complete["resp_id"].nunique()), "complete_n": int(len(tree_complete)),
                     "median_monthly": float(monthly_clean["_monthly"].median()), "correlations": correlations},
            "models": models}


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(build_story_data(), ensure_ascii=False, separators=(",", ":"))
    js_data = f"window.STORY_DATA = {payload};\n"
    (SITE_DIR / "data.js").write_text(js_data, encoding="utf-8")
    (DOCS_DIR / "data.js").write_text(js_data, encoding="utf-8")
    (DOCS_DIR / "styles.css").write_text((SITE_DIR / "styles.css").read_text(encoding="utf-8"), encoding="utf-8")
    body = (SITE_DIR / "_body.qmd").read_text(encoding="utf-8")
    body = body.replace("{{< include ../Story Intro.txt >}}", render_teammate_text())
    head = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="An interactive story about personality and income using TREE and synthetic comparison data.">
<title>The Paycheck Personality</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="styles.css"><script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
</head><body>\n"""
    html = head + body + "\n</body></html>\n"
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    (DOCS_DIR / "article.html").write_text(html, encoding="utf-8")
    print(f"Built GitHub Pages article at {DOCS_DIR / 'article.html'}")


if __name__ == "__main__":
    main()
