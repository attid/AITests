set dotenv-load := true

default:
    @just --list

install:
    uv sync

# Pass extra flags after `--`, e.g.:  just run -- --out runs/xxx --resume
run *args="":
    uv run llm-eval run --config models.yaml --tasks tasks.jsonl {{args}}

report run_dir:
    uv run llm-eval report {{run_dir}}

validate config="models.yaml" tasks="tasks.jsonl":
    uv run llm-eval validate --config {{config}} --tasks {{tasks}}

ping *args="":
    uv run llm-eval ping --config models.yaml {{args}}

test *args="":
    uv run pytest {{args}}

smoke:
    uv run pytest -m smoke

lint:
    uv run ruff check .
    uv run ruff format --check .

format:
    uv run ruff check --fix .
    uv run ruff format .

typecheck:
    uv run pyright

ci: lint typecheck test
