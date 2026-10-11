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
    trait_names = (
        "Openness to Experience",
        "Conscientiousness",
        "Extraversion",
        "Agreeableness",
        "Neuroticism (Emotional Instability)",
    )
    blocks = [block.strip() for block in raw.split("\n\n") if block.strip()]
    rendered = []
    index = 0
    traits_rendered = False
    while index < len(blocks):
        block = blocks[index]
        if not traits_rendered and any(block.startswith(f"{name}:") for name in trait_names):
            cards = []
            while index < len(blocks) and any(blocks[index].startswith(f"{name}:") for name in trait_names):
                trait, description = blocks[index].split(":", 1)
                cards.append(
                    f'<li class="trait-card"><div class="trait-number">{len(cards) + 1:02d}</div>'
                    f"<h3>{html.escape(trait)}</h3><p>{html.escape(description.strip())}</p></li>"
                )
                index += 1
            rendered.append(
                '<div class="trait-intro"><div class="section-tag">The OCEAN framework</div>'
                '<p class="trait-intro-copy">The Big Five describe personality across five complementary spectrums:</p></div>'
                f'<ol class="trait-list">{"".join(cards)}</ol>'
            )
            traits_rendered = True
            continue
        if traits_rendered and any(block.startswith(f"{name}:") for name in trait_names):
            cards = []
            while index < len(blocks) and any(blocks[index].startswith(f"{name}:") for name in trait_names):
                trait, description = blocks[index].split(":", 1)
                cards.append(
                    f'<li class="salary-trait-card"><div class="trait-number">{len(cards) + 1:02d}</div>'
                    f"<h3>{html.escape(trait)}</h3><p>{html.escape(description.strip())}</p></li>"
                )
                index += 1
            rendered.append(
                '<div class="salary-trait-intro"><div class="section-tag">How personality can affect pay</div>'
                '<p class="trait-intro-copy">These are the possible pathways from each personality trait to salary outcomes:</p></div>'
                f'<ol class="trait-list salary-trait-list">{"".join(cards)}</ol>'
            )
            continue
        if block == "Short Summary of What Can Be Seen on the Screen:":
            rendered.append(
                '<div class="inline-visualization">'
                '<div class="section-tag">What the TREE dataset shows</div>'
                '<h3>Personality and hourly wage at career entry</h3>'
                '<p class="visualization-caption">Each bar shows the Pearson correlation between one Big Five trait and log hourly wage. Values close to zero indicate little direct association in this sample.</p>'
                '<div id="tree-inline-chart" class="inline-chart" role="img" aria-label="TREE dataset visualization of personality correlations with log hourly wage"></div>'
                '<div id="tree-inline-table" class="table-wrap"></div>'
                '</div>'
            )
            index += 2
            continue
        if block.startswith("So now that you know what personality is"):
            rendered.append(f"<p>{html.escape(block)}</p>")
            rendered.append(
                '<div class="dataset-overview">'
                '<div class="section-tag">Dataset snapshot</div>'
                '<h3>Personality measured alongside work and pay</h3>'
                '<p class="visualization-caption">The TREE2 analysis links Big Five scores to employment episodes and converts reported pay into a comparable hourly wage.</p>'
                '<div class="dataset-metrics">'
                '<div><strong id="overview-people">—</strong><span>people</span></div>'
                '<div><strong id="overview-episodes">—</strong><span>employment episodes</span></div>'
                '<div><strong id="overview-wage">—</strong><span>median CHF / hour</span></div>'
                '</div>'
                '<div id="overview-wage-chart" class="inline-chart" role="img" aria-label="Distribution of hourly wages in the TREE dataset"></div>'
                '</div>'
            )
            index += 1
            continue
        if block.startswith("Try to see what happens when you add different variables"):
            rendered.append(f"<p>{html.escape(block)}</p>")
            rendered.append(
                '<div class="chart-card narrative-kaggle">'
                '<div class="section-tag">Synthetic Kaggle comparison</div>'
                '<div class="control-row"><label for="kaggle-inline-select">Controls</label><select id="kaggle-inline-select" aria-label="Select synthetic regression controls"></select><span id="kaggle-inline-n" class="source-note"></span></div>'
                '<div class="chart-title"><h3>What changes when we add GPA and cognitive scores?</h3><span>SYNTHETIC DATA · STANDARDIZED OLS</span></div>'
                '<p class="visualization-caption">The coefficients show how the estimated relationship between each personality trait and log income changes as background, education, cognitive ability, grades, and job performance are added.</p>'
                '<div id="kaggle-inline-chart" class="chart" role="img" aria-label="Synthetic Kaggle regression coefficients with selectable controls"></div>'
                '<div id="kaggle-inline-table" class="table-wrap"></div>'
                '</div>'
            )
            index += 1
            continue
        if block == "Kaggle Dataset":
            index += 1
            continue
        if block == "Tree Dataset":
            index += 1
            continue
        if block.startswith("Special thanks to the university of Bern"):
            rendered.append(
                '<section class="article-sources" aria-labelledby="article-sources-title">'
                '<div class="section-tag">Acknowledgements & sources</div>'
                '<h2 id="article-sources-title">Explore the data and research</h2>'
                '<p class="sources-lede">The data and research behind this story are collected here for transparency and further reading.</p>'
                '<div class="source-grid">'
                '<article class="source-card source-card-wide"><div class="source-card-label">Acknowledgement</div>'
                '<p>Special thanks to the University of Bern and especially Sandra Hupka-Brunner for giving us access to the TREE study dataset.</p></article>'
                '<article class="source-card"><div class="source-card-label">TREE study</div>'
                '<p>Learn more about the Swiss TREE study and its research programme.</p>'
                '<a href="https://www.tree.unibe.ch/index_eng.html" target="_blank" rel="noreferrer">Visit the TREE study ↗</a></article>'
                '<article class="source-card"><div class="source-card-label">Synthetic comparison data</div>'
                '<p>The Kaggle dataset used for the illustrative comparison. It may be subject to changes.</p>'
                '<a href="https://www.kaggle.com/datasets/sergionefedov/big-five-personality-and-life-outcomes" target="_blank" rel="noreferrer">View the Kaggle dataset ↗</a></article>'
                '</div>'
                '<div class="reading-list"><div class="source-card-label">Further reading</div>'
                '<ol>'
                '<li><span>Nyhus, E. K. &amp; Pons, E. (2005). <em>The effects of personality on earnings.</em> Journal of Economic Psychology, 26(3), 363–384.</span> <a href="https://doi.org/10.1016/j.joep.2004.07.001" target="_blank" rel="noreferrer">DOI ↗</a></li>'
                '<li><span>Vella, M. (2024). <em>The relationship between the Big Five personality traits and earnings: Evidence from a meta-analysis.</em> Bulletin of Economic Research, 76(3), 685–712.</span> <a href="https://doi.org/10.1111/boer.12437" target="_blank" rel="noreferrer">DOI ↗</a></li>'
                '<li><span>Zell, E. &amp; Lesick, T. L. (2022). <em>Big five personality traits and performance: A quantitative synthesis of 50+ meta-analyses.</em> Journal of Personality, 90(4), 559–573.</span> <a href="https://doi.org/10.1111/jopy.12683" target="_blank" rel="noreferrer">DOI ↗</a></li>'
                '</ol></div></section>'
            )
            index = len(blocks)
            continue
        content = html.escape(block).replace("\n", "<br>\n")
        if block in heading_lines:
            rendered.append(f"<h2>{content}</h2>")
        else:
            rendered.append(f"<p>{content}</p>")
        index += 1
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
    hourly = pd.to_numeric(tree_complete["wage_hourly"], errors="coerce")
    hourly_clean = report_dropna(tree_complete.assign(_hourly=hourly), ["_hourly"], "TREE analysis-sample hourly wage summary")
    counts, edges = np.histogram(hourly_clean["_hourly"], bins=12)

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
                     "median_hourly": float(hourly_clean["_hourly"].median()),
                     "wage_histogram": [{"bin": f"{edges[i]:.0f}–{edges[i + 1]:.0f}", "count": int(counts[i])}
                                        for i in range(len(counts))],
                     "correlations": correlations},
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
