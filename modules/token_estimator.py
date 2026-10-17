"""Session-adaptive, conservative prompt-token estimator (no tokenizer, no remote dependency).

Starts at the fixed conservative 2.8 characters/token. After real Ollama responses (`prompt_eval_count`) it learns the observed
characters-per-token for that model tag and uses the LOWEST ratio seen in a recent window (i.e. the most token-dense prompt) reduced by a safety
margin, clamped to a sane range. Deterministic given the observation history; thread-safe (inference may run on worker threads).
It only changes how much of the (unchanged) configured prompt budget is considered used; the budget itself is never raised.
"""
import threading
from collections import deque

INITIAL_CHARS_PER_TOKEN = 2.8
SAFETY_MARGIN = 0.07  # estimate is made 7% more token-heavy than the densest recent observation
MIN_OBSERVATIONS = 3  # keep the initial conservative value until this many real counts exist
WINDOW = 64
CPT_CLAMP = (2.0, 4.0)


class TokenEstimator:
    def __init__(self, model="", initial=INITIAL_CHARS_PER_TOKEN, margin=SAFETY_MARGIN, min_observations=MIN_OBSERVATIONS, window=WINDOW, clamp=CPT_CLAMP):
        self.model = str(model)
        self.initial = float(initial)
        self.margin = float(margin)
        self.min_observations = int(min_observations)
        self.clamp = clamp
        self._ratios = deque(maxlen=int(window))
        self.observations = 0
        self.estimated_total = 0
        self.actual_total = 0
        self.over_estimates = 0  # calls where the estimate was below the real count (never silent: counted and recorded)
        self._lock = threading.Lock()

    def chars_per_token(self):
        with self._lock:
            return self._cpt_locked()

    def _cpt_locked(self):
        if self.observations < self.min_observations or not self._ratios:
            return self.initial
        lo, hi = self.clamp
        return max(lo, min(hi, min(self._ratios) * (1.0 - self.margin)))

    def estimate(self, text):
        return int(len(text) / self.chars_per_token()) + 1

    def observe(self, chars, actual_tokens, estimated_tokens=None):
        """Record one real (chars, prompt_eval_count) pair; returns the per-call record for telemetry."""
        if not chars or not actual_tokens or int(actual_tokens) <= 0:
            return None
        ratio = float(chars) / float(actual_tokens)
        with self._lock:
            self._ratios.append(ratio)
            self.observations += 1
            self.actual_total += int(actual_tokens)
            if estimated_tokens is not None:
                self.estimated_total += int(estimated_tokens)
                if int(estimated_tokens) < int(actual_tokens):
                    self.over_estimates += 1
        return {"chars": int(chars), "actual_prompt_tokens": int(actual_tokens), "observed_chars_per_token": round(ratio, 4),
                "estimated_prompt_tokens": None if estimated_tokens is None else int(estimated_tokens),
                "estimate_over_actual": None if estimated_tokens is None else round(int(estimated_tokens) / int(actual_tokens), 4)}

    def state(self):
        with self._lock:
            r = list(self._ratios)
            return {"model": self.model, "chars_per_token_in_use": round(self._cpt_locked(), 4), "initial_chars_per_token": self.initial, "safety_margin": self.margin,
                    "observations": self.observations, "min_observed_chars_per_token": round(min(r), 4) if r else None,
                    "mean_observed_chars_per_token": round(sum(r) / len(r), 4) if r else None,
                    "adapted": self.observations >= self.min_observations,
                    "estimate_below_actual_calls": self.over_estimates,
                    "mean_estimate_over_actual": round(self.estimated_total / self.actual_total, 4) if self.actual_total and self.estimated_total else None}
