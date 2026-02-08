"""Utility functions for vLLM server integration."""

import os
import json
import asyncio
import random as _random
import httpx
from typing import Dict, Optional


VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", None)


async def auto_detect_model_name(base_url: str) -> Optional[str]:
    """Query the vLLM /v1/models endpoint and return the first available model name."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                if models:
                    return models[0]["id"]
    except Exception as e:
        print(f"Failed to auto-detect model name from {base_url}: {e}")
    return None


async def make_vllm_request(
    prompt: str,
    base_url: str,
    model_name: str,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 16384,
    stream: bool = True,
    extra_params: Optional[Dict] = None,
    max_retries: int = 1,
) -> Dict:
    """Make a request to a vLLM server's completions endpoint.

    Args:
        prompt: The text prompt to complete
        base_url: vLLM server base URL (e.g. http://localhost:8000/v1)
        model_name: Model name as registered in vLLM
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        stream: Whether to use streaming
        extra_params: Additional parameters (frequency_penalty, etc.)
        max_retries: Number of retries on failure

    Returns:
        Dict with keys: text, finish_reason, usage
    """
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    if extra_params:
        payload.update(extra_params)

    api_url = f"{base_url}/completions"
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            if stream:
                return await _handle_vllm_streaming(api_url, headers, payload)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    api_url, headers=headers, json=payload, timeout=240
                )

                if response.status_code == 500:
                    print(f"vLLM server error (500) on attempt {attempt+1}/{max_retries}")
                    await asyncio.sleep(retry_delay * (2**attempt))
                    continue

                if response.status_code == 429:
                    print(f"vLLM rate limit (429) on attempt {attempt+1}/{max_retries}")
                    await asyncio.sleep(
                        retry_delay * (2**attempt) + _random.uniform(1, 3)
                    )
                    continue

                if response.status_code != 200:
                    print(f"vLLM error: {response.status_code} - {response.text}")
                    if attempt == max_retries - 1:
                        return {
                            "error": f"API error: {response.status_code}",
                            "details": response.text,
                        }
                    await asyncio.sleep(retry_delay * (2**attempt))
                    continue

                result = response.json()
                return {
                    "text": result["choices"][0]["text"],
                    "finish_reason": result["choices"][0].get("finish_reason", ""),
                    "usage": result.get("usage", {}),
                }

        except Exception as e:
            print(f"vLLM request exception (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return {"error": f"Request exception: {str(e)}"}
            await asyncio.sleep(retry_delay * (2**attempt))

    return {"error": "All vLLM request attempts failed"}


async def _handle_vllm_streaming(
    api_url: str, headers: Dict, payload: Dict
) -> Dict:
    """Handle streaming responses from vLLM (OpenAI SSE format)."""
    try:
        collected_text = ""
        finish_reason = None
        usage = None

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", api_url, headers=headers, json=payload, timeout=240
            ) as response:
                if response.status_code != 200:
                    return {
                        "error": f"API error: {response.status_code}",
                        "details": await response.aread(),
                    }

                async for chunk in response.aiter_lines():
                    if not chunk.strip():
                        continue
                    if chunk == "data: [DONE]":
                        break
                    if chunk.startswith("data: "):
                        try:
                            data = json.loads(chunk[6:])
                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]
                                if "text" in choice and choice["text"]:
                                    collected_text += choice["text"]
                                elif "delta" in choice and "content" in choice["delta"]:
                                    collected_text += choice["delta"]["content"]
                                if choice.get("finish_reason"):
                                    finish_reason = choice["finish_reason"]
                            if "usage" in data and data["usage"]:
                                usage = data["usage"]
                        except json.JSONDecodeError:
                            print(f"Failed to parse vLLM chunk: {chunk}")

        return {
            "text": collected_text,
            "finish_reason": finish_reason or "stop",
            "usage": usage or {},
        }

    except Exception as e:
        print(f"vLLM streaming exception: {e}")
        return {"error": f"Streaming exception: {str(e)}"}
