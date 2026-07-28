.PHONY: bootstrap train evaluate error-analysis test report clean

# Uses .venv/bin/python directly rather than `uv run`: pyproject.toml
# declares no [project.dependencies] (deps are pinned in requirements.txt /
# requirements-lock.txt instead, installed via the `uv pip install`
# interface), so `uv run` would treat this as a zero-dependency uv project
# and write a trivial, misleading lockfile on top of the real one. See
# .gitignore.
PY := .venv/bin/python

bootstrap:
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv --python 3.12
	uv pip install -r requirements.txt
	uv pip freeze > requirements-lock.txt
	@echo "bootstrap complete - activate with: source .venv/bin/activate"

train:
	$(PY) -m src.train

evaluate:
	$(PY) -m src.evaluate

error-analysis:
	$(PY) -m src.error_analysis

test:
	$(PY) -m pytest -q

report:
	$(PY) scripts/make_report_assets.py
	$(PY) report/build_pdf.py

clean:
	rm -rf outputs .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
