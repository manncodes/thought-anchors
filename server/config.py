import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATA_DIR: str = "math_rollouts"
    VLLM_BASE_URL: str = "http://localhost:8000/v1"
    VLLM_MODEL_NAME: str = ""
    VLLM_JUDGE_BASE_URL: str = "http://localhost:8001/v1"
    VLLM_JUDGE_MODEL_NAME: str = ""
    HOST: str = "0.0.0.0"
    PORT: int = 8080

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
