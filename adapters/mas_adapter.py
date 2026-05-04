"""Wrap G-Memory's MAS frameworks into EquiMem's interface."""

import sys
sys.path.insert(0, "vendor/GMemory")

from mas.base import MASBase


class MacNetAdapter(MASBase):
    """Wraps G-Memory's MacNet with calibration injection."""

    def __init__(self, backbone, n_agents):
        super().__init__(backbone, n_agents)
        # Import G-Memory's MacNet
        from mas.macnet import MacNetMAS
        self._inner = MacNetMAS(backbone, n_agents)

    def run_round(self, task, memory, safeguard, round_num):
        """Run one debate round with calibration injected."""

        outputs = []
        for agent_idx in range(self.n_agents):
            # --- G-Memory's original agent turn ---
            agent_output = self._inner.run_agent_turn(
                agent_idx, task, memory
            )

            # --- EquiMem injection point ---
            # 1. Extract the proposed delta
            delta = agent_output.proposed_memory_update

            # 2. Collect auditor probes (intercepted queries)
            auditor_probes = []
            for other_idx in range(self.n_agents):
                if other_idx != agent_idx:
                    probe = self._inner.get_last_query(other_idx)
                    auditor_probes.append((f"agent_{other_idx}", probe))

            # 3. Calibrate
            result = safeguard.score(delta, memory.get_state(), auditor_probes)

            # 4. Commit only if calibration passes
            if result.committed:
                memory.save(delta, trust_weight=result.rho)

            outputs.append(agent_output)

        return outputs