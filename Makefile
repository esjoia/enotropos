.PHONY: install dev extract index serve test lint clean

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt
	pip install ruff mypy pytest

extract:
	python winegpt/extract.py --country Espanya

index:
	python scripts/build_index.py --country Espanya

serve:
	python -m streamlit run winegpt/app.py

test:
	python -m pytest tests/ -v

lint:
	python -m ruff check winegpt/ scripts/ tests/

type:
	python -m mypy winegpt/ scripts/ tests/

clean:
	python -c "import shutil; shutil.rmtree('data', ignore_errors=True); \
	shutil.rmtree('__pycache__', ignore_errors=True); shutil.rmtree('.pytest_cache', ignore_errors=True)"
