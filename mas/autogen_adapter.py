"""AutoGen MAS adapter (Wu et al., 2024).

Wraps G-Memory's AutoGen integration. AutoGen uses conversational
turn-taking where agents send messages to each other in sequence.

Same probe interception pattern as MacNet.
"""

import sys
from pathlib import Path

import ollama

from .base import MASBase, AgentOutput
from .macnet_adapter import MacNetAdapter

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "GMemory"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))


class AutoGenAdapter(MacNetAdapter):
    """AutoGen adapter. Shares the same standalone logic as MacNet
    but delegates to G-Memory's AutoGen implementation when available.

    AutoGen's conversational turn-taking naturally produces retrieval
    queries that EquiMem intercepts as probes.
    """

    def __init__(self, backbone: str, n_agents: int):
        MASBase.__init__(self, backbone, n_agents)
        self._history = []
        self._final_answer = ""
        self._vendor = None

        try:
            from mas.autogen import AutoGenMAS
            self._vendor = AutoGenMAS(backbone, n_agents)
        except ImportError:
            pass
