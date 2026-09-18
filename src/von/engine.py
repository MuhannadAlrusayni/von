"""Von Engine orchestrator with pluggable backends (Needle 3, ModernBERT-151M, Qwen-0.5B, Laya-421M)."""

import os
import threading
from typing import Any, Dict, List, Optional, Union

from .backends import (
    BaseBackend,
    LayaBackend,
    ModernBERTBackend,
    NeedleBackend,
    QwenPCDBackend,
)
from .types import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    SystemOneResponse,
)


class VonEngine:
    """System One inference engine orchestrator."""

    _instance: Optional["VonEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, backend_name: str = "needle"):
        self.backend_name = backend_name.lower().strip()
        if self.backend_name in ("needle", "cactus-needle", "needle-json", "needle_json"):
            self.backend: BaseBackend = NeedleBackend()
        elif self.backend_name in ("laya", "laya-421m", "convaiinnovations/laya"):
            self.backend = LayaBackend()
        elif self.backend_name in ("modernbert", "rlcd-modernbert", "bert-151m"):
            self.backend = ModernBERTBackend()
        elif self.backend_name in ("qwen", "qwen0.5b", "qwen-0.5b", "qwen-pcd"):
            self.backend = QwenPCDBackend()
        else:
            raise ValueError(
                f"Unknown backend '{self.backend_name}'. Available: needle, laya, modernbert, qwen0.5b"
            )

    @classmethod
    def get_instance(cls, backend: Optional[str] = None) -> "VonEngine":
        with cls._lock:
            if cls._instance is None:
                b = backend or os.environ.get("VON_BACKEND", "needle")
                cls._instance = cls(backend_name=b)
            return cls._instance

    @classmethod
    def set_backend(cls, backend: str):
        """Switch active engine backend ('needle', 'laya', 'modernbert', 'qwen0.5b')."""
        with cls._lock:
            cls._instance = cls(backend_name=backend)

    def embed(self, text: str) -> List[float]:
        if hasattr(self.backend, "embed"):
            return getattr(self.backend, "embed")(text)
        from .backends.needle_backend import NeedleBackend
        return NeedleBackend().embed(text)

    def evaluate_choice(self, *args, **kwargs) -> ChoiceAnswer:
        return self.backend.evaluate_choice(*args, **kwargs)

    def evaluate_score(self, *args, **kwargs) -> ScoreAnswer:
        return self.backend.evaluate_score(*args, **kwargs)

    def evaluate_noul(self, *args, **kwargs) -> NoulAnswer:
        return self.backend.evaluate_noul(*args, **kwargs)

    def evaluate(
        self,
        state: Any,
        questions: Dict[str, Union[Question, Dict[str, Any]]],
        model: Optional[str] = None,
    ) -> SystemOneResponse:
        if model in ("von-latest", "von-preview", "jev-latest", "jev-preview", None):
            resolved_model = "von-1.0.0"
        else:
            resolved_model = model
        return self.backend.evaluate(state=state, questions=questions, model=resolved_model)
