"""PPL Filter safeguard (Alon et al., 2023).

Rejects memory entries whose perplexity under a reference
language model exceeds an adaptive threshold. No generation
is performed — only a forward pass to compute token-level
log-probabilities.

Token cost: 0 additional generated tokens (forward pass only).
Limitation: adversaries can craft low-perplexity poison
that passes this filter trivially.
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import SafeguardBase
from equimem.calibration.base import CalibrationResult


class PPLFilterSafeguard(SafeguardBase):
    """Filter entries by perplexity under a reference LM."""

    def __init__(
        self,
        reference_model: str = "Qwen/Qwen2.5-1.5B",
        device: str = "cuda",
        window_size: int = 100,
    ):
        self.device = device
        self.window_size = window_size
        self._ppl_history: list[float] = []

        # Load a small reference LM for perplexity computation
        self.tokenizer = AutoTokenizer.from_pretrained(
            reference_model, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            reference_model, trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(device).eval()

    def score(self, delta, memory_state, auditor_signals, **kwargs):
        delta_text = self._serialize_delta(delta)
        ppl = self._compute_perplexity(delta_text)
        threshold = self._adaptive_threshold()

        # Low perplexity = normal text = pass
        # High perplexity = anomalous = reject
        if ppl <= threshold:
            score = 1.0 - (ppl / (threshold + 1e-8)) * 0.3
        else:
            score = max(0.0, 0.3 - (ppl - threshold) / (threshold + 1e-8) * 0.3)

        score = max(0.0, min(1.0, score))
        committed = ppl <= threshold

        # Update history
        self._ppl_history.append(ppl)

        return CalibrationResult(
            rho=score,
            rho_detect=score,
            rho_align=score,
            committed=committed,
            details={"perplexity": ppl, "threshold": threshold},
        )

    def _compute_perplexity(self, text: str) -> float:
        """Compute perplexity via forward pass (no generation)."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        return float(torch.exp(loss).cpu())

    def _adaptive_threshold(self) -> float:
        """Running median of recent perplexity values.

        Same adaptive principle as EquiMem's rho*, but applied
        to perplexity instead of calibration scores.
        """
        if len(self._ppl_history) < 5:
            return 20.0  # cold-start default
        recent = self._ppl_history[-self.window_size:]
        return float(np.median(recent) * 1.5)  # 1.5x median as threshold

    @staticmethod
    def _serialize_delta(delta) -> str:
        if isinstance(delta, str):
            return delta
        if isinstance(delta, list):
            if delta and isinstance(delta[0], tuple):
                if len(delta[0]) == 2:
                    return "\n".join(t for _, t in delta)
                return "\n".join(f"{u} {r} {v}" for u, r, v in delta)
        return str(delta)
