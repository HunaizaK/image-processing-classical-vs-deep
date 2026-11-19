PYTHON ?= python

.PHONY: install all classical cnn robustness report full serve test lint format docker

install:  ## Pinned dependencies + this package (editable)
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

all:  ## Download GTSRB, train both models, robustness tests, report (~4.5 h on CPU)
	$(PYTHON) -m signs all

classical cnn robustness report:
	$(PYTHON) -m signs $@

full:  ## Longer CNN schedule (30 epochs, patience 6), separate output dirs
	$(PYTHON) -m signs all --full

serve:  ## SIGNS_BACKEND=cnn|classical
	uvicorn signs.serve:app --host 0.0.0.0 --port 8000

test:
	$(PYTHON) -m pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

docker:
	docker build -t signs .
