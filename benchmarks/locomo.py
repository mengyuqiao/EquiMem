"""LoCoMo benchmark (Maharana et al., 2024).

Long-term conversational memory QA requiring temporal reasoning
across distant dialogue sessions spanning days to weeks.

Metric: Token-level F1 between predicted and gold answers.

Dataset: loaded from HuggingFace or local JSON.
"""

import re
import string
from collections import Counter
from pathlib import Path

from .base import BenchmarkBase, Task


class LoCoMoBenchmark(BenchmarkBase):
    """LoCoMo conversational memory QA."""

    def __init__(self, data_path: str = "data/locomo", n_tasks: int = 100):
        self.data_path = Path(data_path)
        self.n_tasks = n_tasks
        self._data = None

    def get_tasks(self) -> list[Task]:
        if self._data is None:
            self._data = self._load_data()

        tasks = []
        for i, item in enumerate(self._data):
            if i >= self.n_tasks:
                break

            # Build context from conversation sessions
            context = self._format_conversations(item.get("conversations", []))

            # Each item has multiple QA pairs; we flatten them
            for qa in item.get("qa_pairs", []):
                tasks.append(Task(
                    task_id=f"{item.get('conv_id', i)}_{qa.get('qa_id', '')}",
                    query=qa["question"],
                    context=context,
                    ground_truth=qa["answer"],
                    metadata={
                        "category": qa.get("category", ""),
                        "evidence": qa.get("evidence_ids", []),
                    },
                ))

                if len(tasks) >= self.n_tasks:
                    break

        return tasks[:self.n_tasks]

    def evaluate_answer(self, task: Task, answer: str) -> float:
        """Token-level F1 score."""
        pred_tokens = self._tokenize(answer)
        gold_tokens = self._tokenize(task.ground_truth)

        if not pred_tokens or not gold_tokens:
            return 0.0

        common = Counter(pred_tokens) & Counter(gold_tokens)
        n_common = sum(common.values())

        if n_common == 0:
            return 0.0

        precision = n_common / len(pred_tokens)
        recall = n_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _load_data(self) -> list[dict]:
        """Load LoCoMo dataset from local JSON files."""
        import json

        data = []
        # Try loading from HuggingFace first
        try:
            from datasets import load_dataset
            ds = load_dataset("snap-research/locomo", split="test")
            return list(ds)
        except Exception:
            pass

        # Fallback to local files
        for json_file in sorted(self.data_path.glob("*.json")):
            with open(json_file) as f:
                items = json.load(f)
                if isinstance(items, list):
                    data.extend(items)
                else:
                    data.append(items)

        return data

    @staticmethod
    def _format_conversations(conversations: list) -> str:
        """Format conversation sessions into a context string."""
        parts = []
        for session in conversations:
            if isinstance(session, dict):
                session_text = session.get("text", str(session))
                session_id = session.get("session_id", "")
                parts.append(f"[Session {session_id}]\n{session_text}")
            else:
                parts.append(str(session))
        return "\n\n".join(parts)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Normalize and tokenize for F1 computation."""
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()
