# Quarto article site

`index.qmd` includes the article body in `_body.qmd`. The interactive charts read aggregate statistics from `data.js`; no respondent-level dataset is included in the published site.

Regenerate `data.js` and the HTML article preview from the canonical TREE and synthetic Kaggle data with `python site/build_data.py`. This writes both `docs/index.html` and `docs/article.html`, using the same design and data. Then render the Quarto project with `quarto render site`. The rendered site is written to `docs/` for GitHub Pages.

The workflow in `.github/workflows/publish-quarto.yml` deploys the rendered `docs/` site when changes are pushed to `main` or when manually run from GitHub Actions.
