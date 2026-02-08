import pytest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_vllm_models_response():
    """Mock response from vLLM /v1/models endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": "split-llama-v1",
                "object": "model",
                "created": 1700000000,
                "owned_by": "vllm",
            }
        ],
    }


@pytest.fixture
def mock_vllm_completion_response():
    """Mock non-streaming response from vLLM /v1/completions."""
    return {
        "id": "cmpl-abc123",
        "object": "text_completion",
        "created": 1700000000,
        "model": "split-llama-v1",
        "choices": [
            {
                "index": 0,
                "text": "Let me solve this step by step.\n\nFirst, 2+2=4.\n\nTherefore, \\boxed{4}",
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 30,
            "total_tokens": 80,
        },
    }


@pytest.fixture
def mock_vllm_streaming_chunks():
    """Mock SSE streaming chunks from vLLM."""
    return [
        'data: {"id":"cmpl-1","object":"text_completion","choices":[{"index":0,"text":"Let me ","finish_reason":null}]}',
        'data: {"id":"cmpl-1","object":"text_completion","choices":[{"index":0,"text":"solve this.","finish_reason":null}]}',
        'data: {"id":"cmpl-1","object":"text_completion","choices":[{"index":0,"text":" \\\\boxed{4}","finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}',
        "data: [DONE]",
    ]


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory with sample problem/chunks/solution files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create model directory structure:
        # DATA_DIR/model_name/solution_type/problem_X/
        model_dir = Path(tmpdir) / "split-llama-v1" / "correct_base_solution"
        problem_dir = model_dir / "problem_0"
        problem_dir.mkdir(parents=True)

        # problem.json
        problem_data = {
            "problem": "What is 2+2?",
            "level": "Level 1",
            "type": "Algebra",
            "gt_solution": "2+2=\\boxed{4}",
            "gt_answer": "4",
        }
        with open(problem_dir / "problem.json", "w") as f:
            json.dump(problem_data, f)

        # base_solution.json
        base_solution = {
            "prompt": "Solve this math problem step by step.",
            "solution": "Let me think. 2+2 equals 4. \\boxed{4}",
            "full_cot": "Solve...\n<think>\nLet me think. 2+2 equals 4.\n</think>\n\\boxed{4}",
            "answer": "4",
            "is_correct": True,
        }
        with open(problem_dir / "base_solution.json", "w") as f:
            json.dump(base_solution, f)

        # chunks.json
        chunks_data = {
            "source_text": "Let me think. 2+2 equals 4.",
            "solution_text": "Let me think. 2+2 equals 4.",
            "chunks": ["Let me think.", "2+2 equals 4."],
        }
        with open(problem_dir / "chunks.json", "w") as f:
            json.dump(chunks_data, f)

        # chunks_labeled.json
        chunks_labeled = {
            "0": {
                "function_tags": ["problem_setup"],
                "depends_on": [],
            },
            "1": {
                "function_tags": ["active_computation", "final_answer_emission"],
                "depends_on": ["0"],
            },
        }
        with open(problem_dir / "chunks_labeled.json", "w") as f:
            json.dump(chunks_labeled, f)

        # step_importance.json
        step_importance = {
            "0": {"accuracy_without": 0.3, "accuracy_with": 0.9, "importance": 0.6},
            "1": {"accuracy_without": 0.1, "accuracy_with": 0.9, "importance": 0.8},
        }
        with open(problem_dir / "step_importance.json", "w") as f:
            json.dump(step_importance, f)

        # Create a second problem
        problem_dir_2 = model_dir / "problem_1"
        problem_dir_2.mkdir(parents=True)
        problem_data_2 = {
            "problem": "What is 3*3?",
            "level": "Level 1",
            "type": "Algebra",
            "gt_solution": "3*3=\\boxed{9}",
            "gt_answer": "9",
        }
        with open(problem_dir_2 / "problem.json", "w") as f:
            json.dump(problem_data_2, f)

        yield tmpdir


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.AsyncClient for testing API calls."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "text": "The answer is \\boxed{4}",
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.fixture
def mock_openai_chat_response():
    """Mock response from OpenAI chat completions."""
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(
        {
            "0": {"function_tags": ["problem_setup"], "depends_on": []},
            "1": {
                "function_tags": ["active_computation"],
                "depends_on": ["0"],
            },
        }
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def mock_openai_classify_response():
    """Mock response from OpenAI for classification."""
    mock_choice = MagicMock()
    mock_choice.message.content = "correct"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response
