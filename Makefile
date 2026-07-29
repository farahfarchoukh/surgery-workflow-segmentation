.PHONY: bootstrap bootstrap-dev train evaluate error-analysis test coverage audit hooks-install test report serve docker-build docker-run clean

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
	uv pip install --index-strategy unsafe-best-match -r requirements-lock.txt
	@echo "bootstrap complete - activate with: source .venv/bin/activate"

# Adds pip-audit/pytest-cov/pre-commit on top of bootstrap - kept separate
# so the Docker image (built from requirements-lock.txt alone) never
# carries these dev-only tools. See requirements-dev.txt.
bootstrap-dev: bootstrap
	uv pip install -r requirements-dev.txt

train:
	$(PY) -m src.train

evaluate:
	$(PY) -m src.evaluate

error-analysis:
	$(PY) -m src.error_analysis

test:
	$(PY) -m pytest -q

coverage:
	$(PY) -m pytest -q --cov=src --cov-report=term-missing

audit:
	$(PY) -m pip_audit

hooks-install:
	$(PY) -m pre_commit install

# Deliberately NOT run as part of `bootstrap` - a first version of this
# Makefile ran `uv pip freeze > requirements-lock.txt` on every bootstrap,
# which silently re-resolved and overwrote the checked-in lock file with
# whatever fresh versions existed that day, defeating its own purpose as a
# reproducibility artifact. Regenerating the lock is now this one explicit,
# separate step: a clean throwaway venv (so packages already installed in
# .venv/dev tools like pytest-cov never leak into the runtime lock) resolved
# against requirements.txt's floors, then frozen.
relock:
	uv venv .venv-relock --python 3.12 --clear
	uv pip install --python .venv-relock/bin/python --index-strategy unsafe-best-match -r requirements.txt
	uv pip freeze --python .venv-relock/bin/python > requirements-lock.txt
	@echo "requirements-lock.txt regenerated - review the diff, then re-run 'make bootstrap && make test' before committing"

report:
	$(PY) scripts/make_report_assets.py
	$(PY) report/build_pdf.py
	$(PY) report/build_readme.py

serve:
	$(PY) -m uvicorn src.serve:app --host 127.0.0.1 --port 8000

docker-build:
	docker build -t surgery-workflow-segmentation .

docker-run:
	docker run --rm -it -p 8000:8000 surgery-workflow-segmentation

clean:
	rm -rf outputs .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
