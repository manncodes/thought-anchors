from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class ModelInfo(BaseModel):
    name: str
    solution_types: List[str]


class ProblemSummary(BaseModel):
    problem_id: str
    problem_text: str
    level: Optional[str] = None
    type: Optional[str] = None


class ChunkData(BaseModel):
    function_tags: List[str]
    depends_on: List[str]


class ProblemDetail(BaseModel):
    problem: Dict[str, Any]
    chunks_labeled: Optional[Dict[str, ChunkData]] = None
    step_importance: Optional[Dict[str, Any]] = None
    base_solution: Optional[Dict[str, Any]] = None


class GenerateRequest(BaseModel):
    model: str = "deepseek/deepseek-r1-distill-qwen-14b"
    provider: str = "VLLM"
    num_problems: int = 10
    num_rollouts: int = 100
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 16384
    base_solution_type: str = "correct"
    output_dir: Optional[str] = None


class GenerateResponse(BaseModel):
    status: str
    pid: int
    message: str
