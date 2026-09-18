"""hop - The open-source System One decision model.

Fast, local, non-autoregressive decision primitives.
"""

from .types import (
    Noul,
    Choice,
    Score,
    noul,
    choice,
    score,
    NoulAnswer,
    ChoiceAnswer,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)
from .client import HopClient, AsyncHopClient
from .api import system_one, decide, judge, rate

__version__ = "1.0.0"

__all__ = [
    "Noul",
    "Choice",
    "Score",
    "noul",
    "choice",
    "score",
    "NoulAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "SystemOneResponse",
    "Usage",
    "HopClient",
    "AsyncHopClient",
    "system_one",
    "decide",
    "judge",
    "rate",
]
