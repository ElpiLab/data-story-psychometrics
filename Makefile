.PHONY: install analyze app test site-data site-preview site

install:
	python -m pip install -r requirements.txt

analyze:
	python -m src.tree_analysis
	python -m src.kaggle_analysis
	python -m src.comparison_analysis
	python -m src.advanced_analysis
	python -m src.cleaning_analysis
	python -m src.hourlywage_mean_median

app:
	python -m streamlit run app.py

test:
	python -m pytest

site-data:
	python site/build_data.py

site-preview: site-data
	@echo Open docs/index.html in a browser for the local preview.

site: site-data
	quarto render site
