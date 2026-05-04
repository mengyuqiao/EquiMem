"""Base interface for benchmark evaluators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Task:
    """A single evaluation task."""
    task_id: str
    query: str
    context: str = ""
    ground_truth: str = ""
    metadata: dict = None

    def get_query(self, round_num: int, agent_idx: int) -> str:
        """Return the query for a given round and agent.

        Default: same query every round. Subclasses can override
        for tasks where the query evolves (e.g., ALFWorld actions).
        """
        return self.query


class BenchmarkBase(ABC):
    """All benchmarks implement this interface."""

    @abstractmethod
    def get_tasks(self) -> list[Task]:
        """Load and return all evaluation tasks."""
        ...

    @abstractmethod
    def evaluate_answer(self, task: Task, answer: str) -> float:
        """Score a final answer against the ground truth.

        Returns:
            Score in [0, 1]. 1 = fully correct, 0 = wrong.
        """
        ...
