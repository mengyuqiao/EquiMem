"""Base interface for multi-agent system adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AgentOutput:
    """Output from a single agent turn."""
    agent_idx: int
    response: str
    proposed_delta: object = None  # entries or edges to commit
    tokens_used: int = 0
    retrieval_queries: list[str] = field(default_factory=list)


class MASBase(ABC):
    """Interface for multi-agent debate systems.

    Each adapter wraps a vendor MAS framework (AutoGen, MacNet,
    DyLAN) and exposes a uniform debate-round interface with
    probe interception for EquiMem's calibration.
    """

    def __init__(self, backbone: str, n_agents: int):
        self.backbone = backbone
        self.n_agents = n_agents
        self._last_queries: dict[int, str] = {}
        self._last_paths: dict[int, list] = {}

    @abstractmethod
    def run_agent_turn(self, agent_idx: int, task, context: str,
                       round_num: int) -> AgentOutput:
        """Run a single agent's turn in the debate.

        Args:
            agent_idx: Which agent is acting (contributor).
            task: Current benchmark task.
            context: Retrieved memory context string.
            round_num: Current debate round.

        Returns:
            AgentOutput with response, proposed delta, and token count.
        """
        ...

    @abstractmethod
    def get_final_answer(self) -> str:
        """Return the final consensus answer after all rounds."""
        ...

    def get_intercepted_probe(self, agent_idx: int):
        """Return the last retrieval query or path from this agent.

        This is the probe that EquiMem intercepts for rho_align.
        The agent generates this query during its normal debate
        activity — no additional tokens are produced.
        """
        return self._last_queries.get(agent_idx, "")
