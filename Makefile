.PHONY: install dev-backend dev-frontend test generate-rollouts

install:
	uv pip install -r requirements.txt
	cd frontend && npm install

dev-backend:
	uv run python -m server.main

dev-frontend:
	cd frontend && npm run dev

test:
	uv run pytest tests/ -v

generate-rollouts:
	uv run python generate_rollouts.py -p VLLM -m split-llama $(ARGS)
