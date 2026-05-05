"""DyLAN MAS adapter (Liu et al., 2023).

Wraps G-Memory's DyLAN integration. DyLAN uses a dynamic
LLM-agent network with learned topology selection — agents
are scored by importance and the network adapts which agents
participate in each round.

Same probe interception pattern as MacNet.
"""

import sys
from pathlib import Path

from .base import MASBase, AgentOutput
from .macnet_adapter import MacNetAdapter

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "GMemory"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))


class DyLANAdapter(MacNetAdapter):
    """DyLAN adapter. Shares standalone logic with MacNet but
    delegates to G-Memory's DyLAN implementation when available.

    DyLAN's dynamic topology selection means different agents
    may participate in different rounds, producing diverse
    probes that benefit EquiMem's rho_align computation.
    """

    def __init__(self, backbone: str, n_agents: int):
        MASBase.__init__(self, backbone, n_agents)
        self._history = []
        self._final_answer = ""
        self._vendor = None

        try:
            from mas.dylan import DyLANMAS
            self._vendor = DyLANMAS(backbone, n_agents)
        except ImportError:
            pass
