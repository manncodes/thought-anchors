import pytest
import json
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def test_client(temp_data_dir):
    """Create a FastAPI TestClient with temp data dir."""
    with patch.dict(
        "os.environ",
        {"DATA_DIR": temp_data_dir},
    ):
        from server.config import get_settings
        # Clear any cached settings
        get_settings.cache_clear()

        from server.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        yield client

        get_settings.cache_clear()


class TestListModels:
    """Tests for GET /api/models endpoint."""

    def test_list_models_returns_available_models(self, test_client, temp_data_dir):
        response = test_client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        model_names = [m["name"] for m in data]
        assert "split-llama-v1" in model_names

    def test_list_models_includes_solution_types(self, test_client, temp_data_dir):
        response = test_client.get("/api/models")
        data = response.json()
        model = next(m for m in data if m["name"] == "split-llama-v1")
        assert "solution_types" in model
        assert "correct_base_solution" in model["solution_types"]


class TestListProblems:
    """Tests for GET /api/problems/{model}/{solution_type} endpoint."""

    def test_list_problems_returns_problems(self, test_client):
        response = test_client.get(
            "/api/problems/split-llama-v1/correct_base_solution"
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_list_problems_includes_problem_ids(self, test_client):
        response = test_client.get(
            "/api/problems/split-llama-v1/correct_base_solution"
        )
        data = response.json()
        ids = [p["problem_id"] for p in data]
        assert "problem_0" in ids

    def test_list_problems_nonexistent_model(self, test_client):
        response = test_client.get(
            "/api/problems/nonexistent-model/correct_base_solution"
        )
        assert response.status_code == 404

    def test_list_problems_includes_problem_text(self, test_client):
        response = test_client.get(
            "/api/problems/split-llama-v1/correct_base_solution"
        )
        data = response.json()
        problem = next(p for p in data if p["problem_id"] == "problem_0")
        assert "problem_text" in problem
        assert "2+2" in problem["problem_text"]


class TestGetProblem:
    """Tests for GET /api/problem/{model}/{solution_type}/{problem_id} endpoint."""

    def test_get_problem_returns_full_data(self, test_client):
        response = test_client.get(
            "/api/problem/split-llama-v1/correct_base_solution/problem_0"
        )
        assert response.status_code == 200
        data = response.json()
        assert "problem" in data
        assert "chunks_labeled" in data
        assert "step_importance" in data
        assert "base_solution" in data

    def test_get_problem_correct_content(self, test_client):
        response = test_client.get(
            "/api/problem/split-llama-v1/correct_base_solution/problem_0"
        )
        data = response.json()
        assert data["problem"]["problem"] == "What is 2+2?"
        assert data["base_solution"]["is_correct"] is True

    def test_get_problem_nonexistent(self, test_client):
        response = test_client.get(
            "/api/problem/split-llama-v1/correct_base_solution/problem_999"
        )
        assert response.status_code == 404

    def test_get_problem_missing_optional_files(self, test_client, temp_data_dir):
        """Test that missing optional files (chunks_labeled, step_importance) return null."""
        # problem_1 has only problem.json
        response = test_client.get(
            "/api/problem/split-llama-v1/correct_base_solution/problem_1"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["problem"]["problem"] == "What is 3*3?"
        assert data["chunks_labeled"] is None
        assert data["step_importance"] is None
        assert data["base_solution"] is None


class TestGenerateEndpoint:
    """Tests for POST /api/generate endpoint."""

    def test_generate_returns_accepted(self, test_client):
        with patch("server.routes.subprocess.Popen") as mock_popen:
            mock_popen.return_value.pid = 12345
            response = test_client.post(
                "/api/generate",
                json={
                    "model": "split-llama-v1",
                    "provider": "VLLM",
                    "num_problems": 1,
                    "num_rollouts": 5,
                },
            )
        assert response.status_code == 202
        data = response.json()
        assert "pid" in data
        assert data["status"] == "started"
