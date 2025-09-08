from __future__ import annotations
from collections import deque
from typing import Dict, List
import numpy as np


class PerClassFIFOQuantiles:
    """
    Online per-class quantile estimator using a bounded FIFO buffer.

    - Maintain a fixed-size buffer per class of recent scores (e.g., s = -log p(y|e)).
    - get_tau(class_k, alpha) returns the alpha-quantile of the buffered scores.
    - If buffer is small, falls back to robust defaults.
    """
    def __init__(self, num_classes: int, window_size: int = 4096):
        self.num_classes = int(num_classes)
        self.window_size = int(window_size)
        self.buffers: Dict[int, deque] = {k: deque(maxlen=self.window_size) for k in range(num_classes)}

    def update(self, class_k: int, score: float):
        k = int(class_k)
        self.buffers[k].append(float(score))

    def update_many(self, class_ids: np.ndarray, scores: np.ndarray):
        class_ids = np.asarray(class_ids).astype(int)
        scores = np.asarray(scores).astype(float)
        for k, s in zip(class_ids, scores):
            self.update(int(k), float(s))

    def get_tau(self, class_k: int, alpha: float) -> float:
        """
        Returns the alpha-quantile of the scores for class_k.
        If insufficient data, returns +inf to avoid false acceptance during warmup.
        """
        buf = self.buffers[int(class_k)]
        if len(buf) < 32:
            return float("inf")  # conservative until enough data
        arr = np.fromiter(buf, dtype=float)
        # scores are s = -log p; threshold t_alpha is the alpha-quantile of s
        return float(np.quantile(arr, float(alpha), method="linear"))

    def get_tau_batch(self, class_ids: np.ndarray, alpha: float) -> np.ndarray:
        return np.array([self.get_tau(int(k), alpha) for k in class_ids], dtype=float)

    def size(self, class_k: int) -> int:
        return len(self.buffers[int(class_k)])

    def sizes(self) -> Dict[int, int]:
        return {k: len(buf) for k, buf in self.buffers.items()}

    # Persistence helpers
    def state_dict(self) -> Dict:
        return {
            "num_classes": self.num_classes,
            "window_size": self.window_size,
            "buffers": {k: list(buf) for k, buf in self.buffers.items()},
        }

    def load_state_dict(self, state: Dict):
        self.num_classes = int(state.get("num_classes", self.num_classes))
        self.window_size = int(state.get("window_size", self.window_size))
        self.buffers = {k: deque(state["buffers"].get(k, []), maxlen=self.window_size) for k in range(self.num_classes)}
        return self
