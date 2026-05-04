"""WebShop benchmark (Yao et al., 2022).

Web-based product search and purchase. Agent navigates a
simulated e-commerce website to find and buy a product
matching a natural language instruction.

Metric: Reward score in [0, 1] based on attribute matching
between purchased product and instruction.

Actions: search[query], choose[button]
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from .base import BenchmarkBase, Task


@dataclass
class WebShopTask(Task):
    """Extended task with action history for interactive episodes."""
    instruction: str = ""
    action_history: list = field(default_factory=list)
    current_observation: str = ""
    done: bool = False
    reward: float = 0.0

    def get_query(self, round_num: int, agent_idx: int) -> str:
        """Query evolves with the current observation."""
        if self.current_observation:
            return f"{self.instruction}\n\nCurrent page:\n{self.current_observation}"
        return self.instruction


class WebShopBenchmark(BenchmarkBase):
    """WebShop product search benchmark."""

    def __init__(self, data_path: str = "data/webshop",
                 n_tasks: int = 100, max_actions: int = 20):
        self.data_path = Path(data_path)
        self.n_tasks = n_tasks
        self.max_actions = max_actions
        self._env = None

    def get_tasks(self) -> list[Task]:
        instructions = self._load_instructions()

        tasks = []
        for i, instr in enumerate(instructions):
            if i >= self.n_tasks:
                break
            tasks.append(WebShopTask(
                task_id=f"webshop_{i}",
                query=instr["instruction"],
                instruction=instr["instruction"],
                ground_truth=json.dumps(instr.get("target_attrs", {})),
                metadata=instr,
            ))

        return tasks

    def evaluate_answer(self, task: Task, answer: str) -> float:
        """Compute reward based on attribute matching.

        The answer should be the agent's final purchase action.
        Reward is computed by matching purchased product attributes
        against the instruction requirements.
        """
        if not isinstance(task, WebShopTask):
            return 0.0

        # If the environment tracked the reward during interaction
        if task.reward > 0:
            return task.reward

        # Fallback: parse answer for attribute matching
        return self._compute_reward(task, answer)

    def step(self, task: WebShopTask, action: str) -> dict:
        """Execute an action in the WebShop environment.

        Args:
            task: Current task state.
            action: "search[query]" or "choose[button]"

        Returns:
            dict with keys: observation, reward, done
        """
        if self._env is None:
            self._env = self._init_env()

        result = self._env.step(action)
        task.action_history.append(action)
        task.current_observation = result.get("observation", "")
        task.done = result.get("done", False)
        task.reward = result.get("reward", 0.0)

        return result

    def _load_instructions(self) -> list[dict]:
        """Load WebShop task instructions."""
        # Try the WebShop package first
        try:
            from webshop import WebShopEnv
            env = WebShopEnv()
            return env.get_instructions(self.n_tasks)
        except ImportError:
            pass

        # Fallback to local JSON
        instr_file = self.data_path / "instructions.json"
        if instr_file.exists():
            with open(instr_file) as f:
                return json.load(f)

        # Generate placeholder instructions
        return [
            {"instruction": f"Find a product matching query {i}",
             "target_attrs": {}}
            for i in range(self.n_tasks)
        ]

    def _init_env(self):
        """Initialize the WebShop environment."""
        try:
            from webshop import WebShopEnv
            return WebShopEnv()
        except ImportError:
            raise ImportError(
                "WebShop environment not found. Install via: "
                "pip install webshop  or clone from "
                "https://github.com/princeton-nlp/WebShop"
            )

    @staticmethod
    def _compute_reward(task: WebShopTask, answer: str) -> float:
        """Compute attribute-matching reward.

        Simplified version: checks how many target attributes
        appear in the purchased product description.
        """
        try:
            target = json.loads(task.ground_truth)
        except (json.JSONDecodeError, TypeError):
            return 0.0

        if not target:
            return 0.0

        answer_lower = answer.lower()
        matches = sum(
            1 for attr_val in target.values()
            if str(attr_val).lower() in answer_lower
        )
        return matches / len(target)
