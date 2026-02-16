"""LLM security evaluation: run prompt sets against Ollama/serve, score responses, compute KPIs."""

from ollama_forge.security_eval.client import query_model
from ollama_forge.security_eval.loader import load_prompt_set
from ollama_forge.security_eval.run import run_eval
from ollama_forge.security_eval.scorers import score_extraction, score_refusal

__all__ = [
    "load_prompt_set",
    "query_model",
    "score_refusal",
    "score_extraction",
    "run_eval",
]
