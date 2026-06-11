.PHONY: install test lint ask eval stats clean

PYTHON ?= python3
PKG := src.ragpipe

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m compileall -q src tests
	$(PYTHON) -m ruff check src tests || true

stats:
	$(PYTHON) -m $(PKG).cli --config config.yaml --stats

ask:
	$(PYTHON) -m $(PKG).cli --config config.yaml --query "How does reciprocal rank fusion combine ranked lists?"

eval:
	$(PYTHON) -m $(PKG).cli --config config.yaml --eval --out artifacts/eval.json

clean:
	rm -rf artifacts .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
