"""LLM security evaluation: run prompt sets against Ollama/serve, score responses, compute KPIs."""

from ollama_tools.security_eval.loader import load_prompt_set
from ollama_tools.security_eval.client import query_model
from ollama_tools.security_eval.scorers import score_refusal, score_extraction
from ollama_tools.security_eval.run import run_eval

__all__ = [
    "load_prompt_set",
    "query_model",
    "score_refusal",
    "score_extraction",
    "run_eval",
]
