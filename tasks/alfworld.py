"""ALFWorld benchmark (Shridhar et al., 2020).

Embodied household tasks with sequential action chains.
Agent navigates rooms and manipulates objects to complete
multi-step tasks (e.g., "put a hot mug in the cabinet").

Metric: Success rate (binary: task completed or not).

This adapter wraps G-Memory's existing ALFWorld evaluation
code, which is already functional in vendor/GMemory/.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from .base import BenchmarkBase, Task


@dataclass
class ALFWorldTask(Task):
    """Extended task for interactive ALFWorld episodes."""
    task_type: str = ""
    action_history: list = field(default_factory=list)
    current_observation: str = ""
    done: bool = False
    score: float = 0.0
    max_steps: int = 15

    def get_query(self, round_num: int, agent_idx: int) -> str:
        """Query includes current observation for interactive tasks."""
        if self.current_observation:
            return f"Task: {self.query}\n\nCurrent state:\n{self.current_observation}"
        return self.query


class ALFWorldBenchmark(BenchmarkBase):
    """ALFWorld embodied household tasks.

    Wraps G-Memory's ALFWorld environment. If G-Memory is not
    available, falls back to loading tasks from a JSON file.
    """

    def __init__(self, data_path: str = "data/alfworld",
                 n_tasks: int = 134, split: str = "eval_in_distribution"):
        self.data_path = Path(data_path)
        self.n_tasks = n_tasks
        self.split = split
        self._env = None

    def get_tasks(self) -> list[Task]:
        task_data = self._load_tasks()

        tasks = []
        for i, item in enumerate(task_data):
            if i >= self.n_tasks:
                break
            tasks.append(ALFWorldTask(
                task_id=f"alfworld_{i}",
                query=item.get("task", item.get("desc", "")),
                task_type=item.get("task_type", ""),
                ground_truth="completed",
                metadata=item,
            ))

        return tasks

    def evaluate_answer(self, task: Task, answer: str) -> float:
        """Binary success: 1.0 if task completed, 0.0 otherwise."""
        if isinstance(task, ALFWorldTask):
            return 1.0 if task.score > 0 else 0.0
        # Fallback: check if answer indicates completion
        completion_keywords = ["done", "completed", "success", "finished"]
        return 1.0 if any(k in answer.lower() for k in completion_keywords) else 0.0

    def _load_tasks(self) -> list[dict]:
        """Load ALFWorld tasks."""
        # Try G-Memory's task file first
        task_file = self.data_path / "alfworld_tasks_suffix.json"
        if task_file.exists():
            with open(task_file) as f:
                return json.load(f)

        # Try importing alfworld directly
        try:
            import alfworld.agents.environment as environment
            env = environment.AlfredTWEnv(
                self.split, max_steps=15,
            )
            return [{"task": env.get_task(i)} for i in range(self.n_tasks)]
        except ImportError:
            pass

        raise FileNotFoundError(
            f"ALFWorld tasks not found at {task_file}. "
            f"Download from: https://github.com/alfworld/alfworld "
            f"and place in {self.data_path}/"
        )
