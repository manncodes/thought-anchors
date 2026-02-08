import pytest
import json
import httpx
from unittest.mock import AsyncMock, MagicMock, patch


class TestAutoDetectModelName:
    """Tests for auto_detect_model_name function."""

    @pytest.mark.asyncio
    async def test_auto_detect_returns_first_model(self, mock_vllm_models_response):
        from vllm_utils import auto_detect_model_name

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_vllm_models_response

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("vllm_utils.httpx.AsyncClient", return_value=mock_client):
            model_name = await auto_detect_model_name("http://localhost:8000/v1")

        assert model_name == "split-llama-v1"

    @pytest.mark.asyncio
    async def test_auto_detect_empty_models(self):
        from vllm_utils import auto_detect_model_name

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"object": "list", "data": []}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("vllm_utils.httpx.AsyncClient", return_value=mock_client):
            model_name = await auto_detect_model_name("http://localhost:8000/v1")

        assert model_name is None

    @pytest.mark.asyncio
    async def test_auto_detect_handles_error(self):
        from vllm_utils import auto_detect_model_name

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("vllm_utils.httpx.AsyncClient", return_value=mock_client):
            model_name = await auto_detect_model_name("http://localhost:8000/v1")

        assert model_name is None


class TestMakeVLLMRequest:
    """Tests for make_vllm_request function."""

    @pytest.mark.asyncio
    async def test_non_streaming_request(self, mock_vllm_completion_response):
        from vllm_utils import make_vllm_request

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_vllm_completion_response

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("vllm_utils.httpx.AsyncClient", return_value=mock_client):
            result = await make_vllm_request(
                prompt="Solve this problem",
                base_url="http://localhost:8000/v1",
                model_name="split-llama-v1",
                stream=False,
            )

        assert "text" in result
        assert result["text"] == mock_vllm_completion_response["choices"][0]["text"]
        assert result["finish_reason"] == "stop"
        assert "usage" in result

    @pytest.mark.asyncio
    async def test_streaming_request(self, mock_vllm_streaming_chunks):
        from vllm_utils import make_vllm_request

        async def mock_aiter_lines():
            for chunk in mock_vllm_streaming_chunks:
                yield chunk

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines

        stream_cm = AsyncMock()
        stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        stream_cm.__aexit__ = AsyncMock(return_value=None)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=stream_cm)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("vllm_utils.httpx.AsyncClient", return_value=mock_client):
            result = await make_vllm_request(
                prompt="Solve this problem",
                base_url="http://localhost:8000/v1",
                model_name="split-llama-v1",
                stream=True,
            )

        assert "text" in result
        assert "solve this." in result["text"]
        assert result["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_retries_on_500(self):
        from vllm_utils import make_vllm_request

        mock_error_response = MagicMock()
        mock_error_response.status_code = 500
        mock_error_response.text = "Internal Server Error"

        mock_success_response = MagicMock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {
            "choices": [{"text": "answer", "finish_reason": "stop"}],
            "usage": {},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[mock_error_response, mock_success_response]
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("vllm_utils.httpx.AsyncClient", return_value=mock_client), \
             patch("vllm_utils.asyncio.sleep", new_callable=AsyncMock):
            result = await make_vllm_request(
                prompt="test",
                base_url="http://localhost:8000/v1",
                model_name="split-llama-v1",
                stream=False,
                max_retries=2,
            )

        assert result["text"] == "answer"
        assert result["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_streaming_error_500(self):
        from vllm_utils import _handle_vllm_streaming

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.aread = AsyncMock(return_value=b"Internal Server Error")

        stream_cm = AsyncMock()
        stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        stream_cm.__aexit__ = AsyncMock(return_value=None)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=stream_cm)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("vllm_utils.httpx.AsyncClient", return_value=mock_client):
            result = await _handle_vllm_streaming(
                "http://localhost:8000/v1/completions",
                {"Content-Type": "application/json"},
                {"prompt": "test", "stream": True},
            )

        assert "error" in result

    @pytest.mark.asyncio
    async def test_all_retries_fail(self):
        from vllm_utils import make_vllm_request

        mock_error_response = MagicMock()
        mock_error_response.status_code = 500
        mock_error_response.text = "Internal Server Error"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_error_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("vllm_utils.httpx.AsyncClient", return_value=mock_client), \
             patch("vllm_utils.asyncio.sleep", new_callable=AsyncMock):
            result = await make_vllm_request(
                prompt="test",
                base_url="http://localhost:8000/v1",
                model_name="split-llama-v1",
                stream=False,
                max_retries=2,
            )

        assert "error" in result
