.PHONY: install dev-backend dev-frontend test generate-rollouts

install:
	pip install -r requirements.txt
	cd frontend && npm install

dev-backend:
	python3 -m server.main

dev-frontend:
	cd frontend && npm run dev

test:
	python3 -m pytest tests/ -v

generate-rollouts:
	python3 generate_rollouts.py -p VLLM -m split-llama $(ARGS)
