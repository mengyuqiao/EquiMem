"""Base interface for adversarial attacks."""

from abc import ABC, abstractmethod


class AttackBase(ABC):
    """All attacks implement this interface.

    Attacks modify either the contributor's delta (memory poisoning)
    or the auditor's probe signals (audit collusion), or both.
    """

    def __init__(self, n_adversaries: int):
        self.n_adversaries = n_adversaries

    @abstractmethod
    def modify_delta(self, delta, memory, agent_idx, rng):
        """Modify a contributor's proposed delta.

        Called at step 3 of the debate loop when the contributor
        is an adversarial agent.

        Args:
            delta: Original proposed delta from the agent.
            memory: Current memory state (for crafting plausible poison).
            agent_idx: Index of the adversarial agent.
            rng: numpy random generator for reproducibility.

        Returns:
            Modified (poisoned) delta.
        """
        ...

    def modify_probe(self, probe, delta, memory, rng):
        """Modify an auditor's intercepted probe.

        Called at step 4 of the debate loop when the auditor
        is an adversarial agent. Default: no modification
        (only Audit Collusion overrides this).

        Returns:
            Modified probe.
        """
        return probe  # default: no modification
