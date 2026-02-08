import pytest
import json
from unittest.mock import MagicMock, patch, AsyncMock


class TestJudgeClientInit:
    """Tests for JudgeClient initialization."""

    def test_default_initialization(self):
        with patch("judge_client.OpenAI") as mock_openai:
            from judge_client import JudgeClient

            client = JudgeClient()
            assert client.api_key == "EMPTY"

    def test_custom_base_url(self):
        with patch("judge_client.OpenAI") as mock_openai:
            from judge_client import JudgeClient

            client = JudgeClient(base_url="http://custom:9000/v1")
            mock_openai.assert_called_once_with(
                base_url="http://custom:9000/v1", api_key="EMPTY"
            )

    def test_custom_api_key(self):
        with patch("judge_client.OpenAI") as mock_openai:
            from judge_client import JudgeClient

            client = JudgeClient(api_key="my-key")
            mock_openai.assert_called_once_with(
                base_url="http://localhost:8001/v1", api_key="my-key"
            )

    def test_model_name_provided(self):
        with patch("judge_client.OpenAI") as mock_openai:
            from judge_client import JudgeClient

            client = JudgeClient(model_name="gpt-oss-judge")
            assert client.model_name == "gpt-oss-judge"

    def test_model_name_auto_detect(self, mock_vllm_models_response):
        mock_models = MagicMock()
        mock_models.list.return_value = MagicMock(
            data=[MagicMock(id="gpt-oss-judge")]
        )
        mock_openai_instance = MagicMock()
        mock_openai_instance.models = mock_models

        with patch("judge_client.OpenAI", return_value=mock_openai_instance):
            from judge_client import JudgeClient

            client = JudgeClient()
            assert client.model_name == "gpt-oss-judge"


class TestLabelChunks:
    """Tests for JudgeClient.label_chunks method."""

    def test_label_chunks_returns_parsed_json(self, mock_openai_chat_response):
        mock_completions = MagicMock()
        mock_completions.create.return_value = mock_openai_chat_response

        mock_chat = MagicMock()
        mock_chat.completions = mock_completions

        mock_openai_instance = MagicMock()
        mock_openai_instance.chat = mock_chat
        mock_openai_instance.models = MagicMock()
        mock_openai_instance.models.list.return_value = MagicMock(
            data=[MagicMock(id="judge-model")]
        )

        with patch("judge_client.OpenAI", return_value=mock_openai_instance):
            from judge_client import JudgeClient

            client = JudgeClient(model_name="judge-model")
            result = client.label_chunks(
                "What is 2+2?",
                ["Let me think.", "2+2=4"],
            )

        assert "0" in result
        assert "function_tags" in result["0"]
        assert result["0"]["function_tags"] == ["problem_setup"]
        assert result["1"]["depends_on"] == ["0"]

    def test_label_chunks_sends_dag_prompt(self):
        mock_completions = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(
            {"0": {"function_tags": ["unknown"], "depends_on": []}}
        )
        mock_completions.create.return_value = MagicMock(choices=[mock_choice])

        mock_chat = MagicMock()
        mock_chat.completions = mock_completions

        mock_openai_instance = MagicMock()
        mock_openai_instance.chat = mock_chat
        mock_openai_instance.models = MagicMock()
        mock_openai_instance.models.list.return_value = MagicMock(
            data=[MagicMock(id="judge-model")]
        )

        with patch("judge_client.OpenAI", return_value=mock_openai_instance):
            from judge_client import JudgeClient

            client = JudgeClient(model_name="judge-model")
            client.label_chunks("problem text", ["chunk1"])

        call_args = mock_completions.create.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        user_msg = [m for m in messages if m["role"] == "user"][0]
        assert "problem text" in user_msg["content"]
        assert "chunk1" in user_msg["content"]


class TestClassifyAnswer:
    """Tests for JudgeClient.classify_answer method."""

    def test_classify_answer_correct(self, mock_openai_classify_response):
        mock_completions = MagicMock()
        mock_completions.create.return_value = mock_openai_classify_response

        mock_chat = MagicMock()
        mock_chat.completions = mock_completions

        mock_openai_instance = MagicMock()
        mock_openai_instance.chat = mock_chat
        mock_openai_instance.models = MagicMock()
        mock_openai_instance.models.list.return_value = MagicMock(
            data=[MagicMock(id="judge-model")]
        )

        with patch("judge_client.OpenAI", return_value=mock_openai_instance):
            from judge_client import JudgeClient

            client = JudgeClient(model_name="judge-model")
            result = client.classify_answer("What is 2+2?", "4")

        assert result == "correct"

    def test_classify_answer_incorrect(self):
        mock_choice = MagicMock()
        mock_choice.message.content = "incorrect"
        mock_completions = MagicMock()
        mock_completions.create.return_value = MagicMock(choices=[mock_choice])

        mock_chat = MagicMock()
        mock_chat.completions = mock_completions

        mock_openai_instance = MagicMock()
        mock_openai_instance.chat = mock_chat
        mock_openai_instance.models = MagicMock()
        mock_openai_instance.models.list.return_value = MagicMock(
            data=[MagicMock(id="judge-model")]
        )

        with patch("judge_client.OpenAI", return_value=mock_openai_instance):
            from judge_client import JudgeClient

            client = JudgeClient(model_name="judge-model")
            result = client.classify_answer("What is 2+2?", "5")

        assert result == "incorrect"


class TestClassifyRollout:
    """Tests for JudgeClient.classify_rollout method."""

    def test_classify_rollout(self):
        mock_choice = MagicMock()
        mock_choice.message.content = "correct"
        mock_completions = MagicMock()
        mock_completions.create.return_value = MagicMock(choices=[mock_choice])

        mock_chat = MagicMock()
        mock_chat.completions = mock_completions

        mock_openai_instance = MagicMock()
        mock_openai_instance.chat = mock_chat
        mock_openai_instance.models = MagicMock()
        mock_openai_instance.models.list.return_value = MagicMock(
            data=[MagicMock(id="judge-model")]
        )

        with patch("judge_client.OpenAI", return_value=mock_openai_instance):
            from judge_client import JudgeClient

            client = JudgeClient(model_name="judge-model")
            result = client.classify_rollout(
                "What is 2+2?",
                "Let me think step by step. 2+2=4. \\boxed{4}",
            )

        assert result == "correct"
