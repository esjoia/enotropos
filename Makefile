.PHONY: install dev local extract index index-knowledge graph tables serve test lint type clean

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt
	pip install ruff mypy pytest

local:
	pip install -r requirements.txt
	pip install ".[local]"

extract:
	python -m winegpt.extract --country Espanya

index:
	python -m scripts.build_index --country Espanya

index-knowledge:
	python -m scripts.build_knowledge_index

graph:
	python -m scripts.build_graph

tables:
	python -m scripts.build_tables_db

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
