"""MacNet MAS adapter (Qian et al., 2024).

Wraps G-Memory's MacNet implementation. MacNet uses DAG-based
topological ordering for agent communication — agents interact
in pairs along edges, passing refined artifacts.

Key adaptation: intercepts each agent's retrieval query after
generation and exposes it via get_intercepted_probe() for
EquiMem's rho_align computation.
"""

import sys
from pathlib import Path

import ollama

from .base import MASBase, AgentOutput

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "GMemory"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))


AGENT_PROMPT = """You are Agent {agent_idx} in a {n_agents}-agent debate system.
Your task: {task_query}

Memory context (retrieved from shared memory):
---
{context}
---

Previous debate history this round:
---
{history}
---

Instructions:
1. Reason about the task using the context and history.
2. If you have new information to add to shared memory, write it
   after "MEMORY UPDATE:" on a new line.
3. Provide your current answer after "ANSWER:" on a new line.
4. Write a retrieval query for the next round after "QUERY:" on a new line.

Your response:"""


class MacNetAdapter(MASBase):
    """MacNet with EquiMem probe interception."""

    def __init__(self, backbone: str, n_agents: int):
        super().__init__(backbone, n_agents)
        self._history: list[str] = []
        self._final_answer: str = ""
        self._vendor = None

        try:
            from mas.macnet import MacNetMAS
            self._vendor = MacNetMAS(backbone, n_agents)
        except ImportError:
            pass

    def run_agent_turn(self, agent_idx, task, context, round_num):
        if self._vendor is not None:
            return self._run_vendor(agent_idx, task, context, round_num)
        return self._run_standalone(agent_idx, task, context, round_num)

    def get_final_answer(self) -> str:
        return self._final_answer

    def _run_standalone(self, agent_idx, task, context, round_num):
        """Standalone mode when vendor is not available."""
        task_query = task.query if hasattr(task, "query") else str(task)
        history = "\n".join(self._history[-5:])

        prompt = AGENT_PROMPT.format(
            agent_idx=agent_idx,
            n_agents=self.n_agents,
            task_query=task_query,
            context=context if context else "(empty)",
            history=history if history else "(none)",
        )

        response = ollama.chat(
            model=self.backbone,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7, "num_predict": 512},
        )

        raw = response.get("message", {}).get("content", "")
        tokens = response.get("eval_count", 0) + response.get("prompt_eval_count", 0)

        # Parse structured output
        delta = self._extract_section(raw, "MEMORY UPDATE:")
        answer = self._extract_section(raw, "ANSWER:")
        query = self._extract_section(raw, "QUERY:")

        # Store intercepted probe
        self._last_queries[agent_idx] = query if query else task_query

        # Update history and final answer
        self._history.append(f"[Agent {agent_idx}] {raw[:200]}")
        if answer:
            self._final_answer = answer

        return AgentOutput(
            agent_idx=agent_idx,
            response=raw,
            proposed_delta=delta,
            tokens_used=tokens,
            retrieval_queries=[query] if query else [],
        )

    def _run_vendor(self, agent_idx, task, context, round_num):
        """Use G-Memory's MacNet implementation."""
        output = self._vendor.run_agent_turn(agent_idx, task, context, round_num)

        # Intercept the retrieval query
        if hasattr(output, "retrieval_queries") and output.retrieval_queries:
            self._last_queries[agent_idx] = output.retrieval_queries[-1]
        elif hasattr(self._vendor, "get_last_query"):
            self._last_queries[agent_idx] = self._vendor.get_last_query(agent_idx)

        return output

    @staticmethod
    def _extract_section(text: str, marker: str) -> str:
        """Extract text after a marker line."""
        if marker not in text:
            return ""
        after = text.split(marker, 1)[1]
        # Take until next marker or end
        for next_marker in ["MEMORY UPDATE:", "ANSWER:", "QUERY:"]:
            if next_marker != marker and next_marker in after:
                after = after.split(next_marker, 1)[0]
        return after.strip()
