import os
import json
from openai import OpenAI
from prompts import DAG_PROMPT


class JudgeClient:
    """Client for interacting with a vLLM-hosted judge model via OpenAI-compatible API."""

    def __init__(self, base_url=None, model_name=None, api_key="EMPTY"):
        self.base_url = base_url or os.getenv(
            "VLLM_JUDGE_BASE_URL", "http://localhost:8001/v1"
        )
        self.api_key = api_key
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        if model_name:
            self.model_name = model_name
        else:
            self.model_name = os.getenv("VLLM_JUDGE_MODEL_NAME") or self._auto_detect_model()

    def _auto_detect_model(self):
        """Query /v1/models and return the first available model name."""
        try:
            models = self.client.models.list()
            if models.data:
                return models.data[0].id
        except Exception as e:
            print(f"Failed to auto-detect judge model: {e}")
        return None

    def label_chunks(self, problem_text, chunks):
        """Send the DAG_PROMPT to the judge model to label chunks with function tags and dependencies.

        Args:
            problem_text: The math problem text.
            chunks: List of chunk strings from the CoT.

        Returns:
            Dict mapping chunk index to function_tags and depends_on.
        """
        full_chunked_text = "\n".join(
            f"[Chunk {i}]: {chunk}" for i, chunk in enumerate(chunks)
        )
        prompt = DAG_PROMPT.format(
            problem_text=problem_text,
            full_chunked_text=full_chunked_text,
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing chain-of-thought reasoning traces."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        content = response.choices[0].message.content

        # Try to parse JSON from the response (handle markdown code blocks)
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (code block markers)
            content = "\n".join(lines[1:-1])

        return json.loads(content)

    def classify_answer(self, problem, answer):
        """Check if an answer to a problem is correct using the judge model.

        Args:
            problem: The problem text.
            answer: The answer to classify.

        Returns:
            String "correct" or "incorrect".
        """
        prompt = (
            f"Given the following math problem and answer, determine if the answer is correct.\n\n"
            f"Problem: {problem}\n"
            f"Answer: {answer}\n\n"
            f"Respond with exactly one word: 'correct' or 'incorrect'."
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a math answer verification expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        return response.choices[0].message.content.strip().lower()

    def classify_rollout(self, problem, rollout_text):
        """Classify whether a rollout produces a correct answer.

        Args:
            problem: The problem text.
            rollout_text: The full rollout text including reasoning.

        Returns:
            String "correct" or "incorrect".
        """
        prompt = (
            f"Given the following math problem and solution attempt, determine if the final answer is correct.\n\n"
            f"Problem: {problem}\n"
            f"Solution attempt:\n{rollout_text}\n\n"
            f"Respond with exactly one word: 'correct' or 'incorrect'."
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a math answer verification expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        return response.choices[0].message.content.strip().lower()
