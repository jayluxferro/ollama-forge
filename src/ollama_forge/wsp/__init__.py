"""Weight-Space Programming (wsp) toolkit.

Phase 1: behavioural diagnostics — measure obstruction signals via API calls.
Phase 2: weight analysis — locate obstruction directions in MLX tensors.
Phase 3: surgical editing — remove obstructions with minimal weight budget.
"""

from ollama_forge.wsp.analyzer import LayerAnalysis, WeightMap, WeightSpaceAnalyzer
from ollama_forge.wsp.diagnostics import (
    ObstructionReport,
    run_diagnostics,
)
from ollama_forge.wsp.sharpener import EditRecord, Sharpener, SharpenResult

__all__ = [
    "ObstructionReport",
    "run_diagnostics",
    "LayerAnalysis",
    "WeightMap",
    "WeightSpaceAnalyzer",
    "EditRecord",
    "SharpenResult",
    "Sharpener",
]
