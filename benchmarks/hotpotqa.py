"""HotpotQA benchmark (Yang et al., 2018).

Multi-hop question answering over Wikipedia paragraphs.
Each question requires reasoning across 2+ documents.

Metric: Exact match accuracy (ground truth must appear
in the agent's final response).

Dataset: loaded via HuggingFace datasets.
Setting: distractor (2 gold + 8 distractor paragraphs).
"""

import re
import string
from datasets import load_dataset
from .base import BenchmarkBase, Task


class HotpotQABenchmark(BenchmarkBase):
    """HotpotQA distractor setting."""

    def __init__(self, n_tasks: int = 100, split: str = "validation"):
        self.n_tasks = n_tasks
        self.split = split
        self._dataset = None

    def get_tasks(self) -> list[Task]:
        if self._dataset is None:
            self._dataset = load_dataset(
                "hotpot_qa", "distractor", split=self.split
            )

        tasks = []
        for i, item in enumerate(self._dataset):
            if i >= self.n_tasks:
                break

            # Build context from supporting paragraphs
            context_parts = []
            for title, sentences in zip(item["context"]["title"],
                                         item["context"]["sentences"]):
                paragraph = " ".join(sentences)
                context_parts.append(f"[{title}] {paragraph}")
            context = "\n\n".join(context_parts)

            tasks.append(Task(
                task_id=item["id"],
                query=item["question"],
                context=context,
                ground_truth=item["answer"],
                metadata={
                    "type": item.get("type", ""),
                    "level": item.get("level", ""),
                },
            ))

        return tasks

    def evaluate_answer(self, task: Task, answer: str) -> float:
        """Exact match: normalized ground truth in normalized answer."""
        pred = self._normalize(answer)
        gold = self._normalize(task.ground_truth)

        if gold in pred:
            return 1.0
        return 0.0

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase, remove articles/punctuation, collapse whitespace."""
        text = text.lower()
        # Remove articles
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Collapse whitespace
        text = " ".join(text.split())
        return text.strip()
