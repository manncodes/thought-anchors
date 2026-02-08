import json
import subprocess
import sys
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Depends

from server.config import Settings, get_settings
from server.schemas import (
    ModelInfo,
    ProblemSummary,
    ProblemDetail,
    GenerateRequest,
    GenerateResponse,
)

router = APIRouter(prefix="/api")


@router.get("/models", response_model=List[ModelInfo])
def list_models(settings: Settings = Depends(get_settings)):
    """List available models by scanning DATA_DIR subdirectories."""
    data_dir = Path(settings.DATA_DIR)
    if not data_dir.exists():
        return []

    models = []
    for model_dir in sorted(data_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        # Each model dir can have solution_type subdirs
        solution_types = [
            d.name
            for d in sorted(model_dir.iterdir())
            if d.is_dir() and not d.name.startswith(".")
        ]
        if solution_types:
            models.append(ModelInfo(name=model_dir.name, solution_types=solution_types))
    return models


@router.get("/problems/{model}/{solution_type}", response_model=List[ProblemSummary])
def list_problems(
    model: str,
    solution_type: str,
    settings: Settings = Depends(get_settings),
):
    """List problems for a given model and solution type."""
    problems_dir = Path(settings.DATA_DIR) / model / solution_type
    if not problems_dir.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {model}/{solution_type}")

    problems = []
    for problem_dir in sorted(problems_dir.iterdir()):
        if not problem_dir.is_dir() or not problem_dir.name.startswith("problem_"):
            continue
        problem_file = problem_dir / "problem.json"
        if not problem_file.exists():
            continue
        with open(problem_file, "r") as f:
            problem_data = json.load(f)
        problems.append(
            ProblemSummary(
                problem_id=problem_dir.name,
                problem_text=problem_data.get("problem", ""),
                level=problem_data.get("level"),
                type=problem_data.get("type"),
            )
        )
    return problems


@router.get("/problem/{model}/{solution_type}/{problem_id}", response_model=ProblemDetail)
def get_problem(
    model: str,
    solution_type: str,
    problem_id: str,
    settings: Settings = Depends(get_settings),
):
    """Return full problem data including labeled chunks and step importance."""
    problem_dir = Path(settings.DATA_DIR) / model / solution_type / problem_id
    if not problem_dir.exists():
        raise HTTPException(status_code=404, detail=f"Problem not found: {problem_id}")

    problem_file = problem_dir / "problem.json"
    if not problem_file.exists():
        raise HTTPException(status_code=404, detail=f"problem.json not found for {problem_id}")

    with open(problem_file, "r") as f:
        problem_data = json.load(f)

    # Load optional files
    chunks_labeled = None
    chunks_labeled_file = problem_dir / "chunks_labeled.json"
    if chunks_labeled_file.exists():
        with open(chunks_labeled_file, "r") as f:
            chunks_labeled = json.load(f)

    step_importance = None
    step_importance_file = problem_dir / "step_importance.json"
    if step_importance_file.exists():
        with open(step_importance_file, "r") as f:
            step_importance = json.load(f)

    base_solution = None
    base_solution_file = problem_dir / "base_solution.json"
    if base_solution_file.exists():
        with open(base_solution_file, "r") as f:
            base_solution = json.load(f)

    return ProblemDetail(
        problem=problem_data,
        chunks_labeled=chunks_labeled,
        step_importance=step_importance,
        base_solution=base_solution,
    )


@router.post("/generate", response_model=GenerateResponse, status_code=202)
def trigger_generate(request: GenerateRequest):
    """Trigger rollout generation by running generate_rollouts.py as a subprocess."""
    cmd = [
        sys.executable,
        "generate_rollouts.py",
        "-m", request.model,
        "-p", request.provider,
        "-np", str(request.num_problems),
        "-nr", str(request.num_rollouts),
        "-t", str(request.temperature),
        "-tp", str(request.top_p),
        "-mt", str(request.max_tokens),
        "-b", request.base_solution_type,
    ]
    if request.output_dir:
        cmd.extend(["-o", request.output_dir])

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return GenerateResponse(
        status="started",
        pid=process.pid,
        message=f"Generation started with PID {process.pid}",
    )
