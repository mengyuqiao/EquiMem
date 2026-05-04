"""Main entry point for EquiMem experiments.

Wraps G-Memory's codebase (vendor/GMemory) via thin adapters,
injecting EquiMem's calibration into the debate loop.

Usage:
    # Basic run
    python tasks/run.py \
        --benchmark hotpotqa --mas macnet --memory gmemory \
        --safeguard equimem --backbone qwen3:8b --n_agents 6

    # Adversarial evaluation
    python tasks/run.py \
        --benchmark hotpotqa --mas macnet --memory memorybank \
        --safeguard equimem --attack memory_poisoning --n_adversaries 3

    # Ablation
    python tasks/run.py \
        --benchmark hotpotqa --mas macnet --memory memorybank \
        --safeguard equimem --ablation wo_align
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure vendor code is importable
VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "GMemory"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

logger = logging.getLogger("equimem")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class RunMetrics:
    """Metrics collected during a single run."""
    score: float = 0.0
    total_tokens: int = 0
    agent_time_s: float = 0.0
    memory_time_s: float = 0.0
    calibration_time_s: float = 0.0
    n_deltas_proposed: int = 0
    n_deltas_committed: int = 0
    n_deltas_rejected: int = 0
    per_round: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="EquiMem: Zero-Trust Memory Calibration for MAD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--benchmark", required=True,
                   choices=["hotpotqa", "locomo", "alfworld", "webshop"])
    p.add_argument("--mas", required=True,
                   choices=["autogen", "macnet", "dylan"])
    p.add_argument("--memory", required=True,
                   choices=["none", "voyager", "memorybank", "arigraph", "gmemory"])
    p.add_argument("--safeguard", required=True,
                   choices=["vanilla", "llm_audit", "ppl_filter", "a_memguard", "equimem"])
    p.add_argument("--backbone", default="qwen3:8b",
                   help="Ollama model name for agent backbone")
    p.add_argument("--n_agents", type=int, default=6)
    p.add_argument("--n_runs", type=int, default=5)
    p.add_argument("--max_rounds", type=int, default=5)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--output_dir", default="results/")

    # Ablation
    p.add_argument("--ablation", default=None,
                   choices=["wo_detect", "wo_align", "wo_reweight"],
                   help="Disable one calibration component")

    # Adversarial attack
    p.add_argument("--attack", default=None,
                   choices=["memory_poisoning", "audit_collusion"])
    p.add_argument("--n_adversaries", type=int, default=0)

    # Misc
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    p.add_argument("--save_trajectory", action="store_true",
                   help="Save full agent trajectories")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Component loaders
# ---------------------------------------------------------------------------
def is_graph_memory(name: str) -> bool:
    return name in ("arigraph", "gmemory")


def load_memory(name, backbone):
    if name == "none":
        return None
    if name == "voyager":
        from adapters.memory_adapter import VoyagerAdapter
        return VoyagerAdapter([backbone])
    if name == "memorybank":
        from adapters.memory_adapter import MemoryBankAdapter
        return MemoryBankAdapter([backbone])
    if name == "arigraph":
        from adapters.memory_adapter import AriGraphAdapter
        return AriGraphAdapter([backbone])
    if name == "gmemory":
        from adapters.memory_adapter import GMemoryAdapter
        return GMemoryAdapter([backbone])
    raise ValueError(f"Unknown memory: {name}")


def load_safeguard(name, memory_type, ablation=None):
    if name == "vanilla":
        from baselines.vanilla import VanillaSafeguard
        return VanillaSafeguard()
    if name == "llm_audit":
        from baselines.llm_audit import LLMAuditSafeguard
        return LLMAuditSafeguard()
    if name == "ppl_filter":
        from baselines.ppl_filter import PPLFilterSafeguard
        return PPLFilterSafeguard()
    if name == "a_memguard":
        if is_graph_memory(memory_type):
            raise ValueError("A-MemGuard incompatible with graph memory")
        from baselines.a_memguard import AMemGuardSafeguard
        return AMemGuardSafeguard()
    if name == "equimem":
        if is_graph_memory(memory_type):
            from equimem.calibration.graph import GraphCalibrator
            cal = GraphCalibrator()
        else:
            from equimem.calibration.embedding import EmbeddingCalibrator
            cal = EmbeddingCalibrator()

        if ablation == "wo_detect":
            cal.disable_detect = True
        elif ablation == "wo_align":
            cal.disable_align = True
        elif ablation == "wo_reweight":
            cal.disable_reweight = True

        return cal
    raise ValueError(f"Unknown safeguard: {name}")


def load_mas(name, backbone, n_agents):
    if name == "autogen":
        from adapters.mas_adapter import AutoGenAdapter
        return AutoGenAdapter(backbone, n_agents)
    if name == "macnet":
        from adapters.mas_adapter import MacNetAdapter
        return MacNetAdapter(backbone, n_agents)
    if name == "dylan":
        from adapters.mas_adapter import DyLANAdapter
        return DyLANAdapter(backbone, n_agents)
    raise ValueError(f"Unknown MAS: {name}")


def load_benchmark(name):
    if name == "hotpotqa":
        from benchmarks.hotpotqa import HotpotQABenchmark
        return HotpotQABenchmark()
    if name == "locomo":
        from benchmarks.locomo import LoCoMoBenchmark
        return LoCoMoBenchmark()
    if name == "alfworld":
        from benchmarks.alfworld import ALFWorldBenchmark
        return ALFWorldBenchmark()
    if name == "webshop":
        from benchmarks.webshop import WebShopBenchmark
        return WebShopBenchmark()
    raise ValueError(f"Unknown benchmark: {name}")


def load_attack(name, n_adversaries, n_agents):
    if name is None or n_adversaries == 0:
        return None
    if n_adversaries >= n_agents:
        raise ValueError(f"n_adversaries ({n_adversaries}) >= n_agents ({n_agents})")
    if name == "memory_poisoning":
        from attacks.memory_poisoning import MemoryPoisoningAttack
        return MemoryPoisoningAttack(n_adversaries)
    if name == "audit_collusion":
        from attacks.audit_collusion import AuditCollusionAttack
        return AuditCollusionAttack(n_adversaries)
    raise ValueError(f"Unknown attack: {name}")


# ---------------------------------------------------------------------------
# Core debate loop with calibration injection
# ---------------------------------------------------------------------------
def run_single(benchmark, mas, memory, safeguard, attack,
               max_rounds, seed, save_trajectory):
    """Execute one full evaluation run.

    Core loop per task:
      for each round:
        for each agent (contributor):
          1. Retrieve from memory                 → memory_time
          2. Generate response + proposed delta    → agent_time
          3. [Attack modifies delta/probes]
          4. Intercept auditor probes
          5. safeguard.score(delta, M_t, probes)   → calibration_time
          6. If committed: memory.save(delta, rho) → memory_time
    """
    rng = np.random.default_rng(seed)
    metrics = RunMetrics()
    trajectories = [] if save_trajectory else None

    # Select adversary agents (fixed for the run)
    adversary_ids = set()
    if attack is not None:
        all_ids = list(range(mas.n_agents))
        adversary_ids = set(
            rng.choice(all_ids, size=attack.n_adversaries, replace=False)
        )
        logger.info(f"Adversary agents: {sorted(adversary_ids)}")

    tasks = benchmark.get_tasks()

    for task_idx, task in enumerate(tasks):
        logger.debug(f"Task {task_idx+1}/{len(tasks)}")

        if memory is not None:
            memory.reset()

        task_traj = {"task": str(task), "rounds": []} if save_trajectory else None

        for round_num in range(max_rounds):
            round_data = []

            for agent_idx in range(mas.n_agents):
                # ---- 1. Memory retrieval ----
                t0 = time.perf_counter()
                context = ""
                if memory is not None:
                    query = task.get_query(round_num, agent_idx)
                    context = memory.search(query, top_k=5)
                metrics.memory_time_s += time.perf_counter() - t0

                # ---- 2. Agent turn ----
                t0 = time.perf_counter()
                output = mas.run_agent_turn(
                    agent_idx=agent_idx,
                    task=task,
                    context=context,
                    round_num=round_num,
                )
                metrics.agent_time_s += time.perf_counter() - t0
                metrics.total_tokens += output.tokens_used

                delta = output.proposed_delta
                if delta is None:
                    continue
                metrics.n_deltas_proposed += 1

                # ---- 3. Attack: modify contributor delta ----
                if attack and agent_idx in adversary_ids:
                    delta = attack.modify_delta(delta, memory, agent_idx, rng)

                # ---- 4. Intercept auditor probes ----
                auditor_probes = []
                for other_idx in range(mas.n_agents):
                    if other_idx == agent_idx:
                        continue
                    probe = mas.get_intercepted_probe(other_idx)

                    # Attack: adversarial auditors craft probes
                    if (attack and other_idx in adversary_ids
                            and hasattr(attack, "modify_probe")):
                        probe = attack.modify_probe(probe, delta, memory, rng)

                    auditor_probes.append((f"agent_{other_idx}", probe))

                # ---- 5. Calibrate ----
                t0 = time.perf_counter()
                result = safeguard.score(
                    delta=delta,
                    memory_state=memory.get_state() if memory else None,
                    auditor_signals=auditor_probes,
                )
                metrics.calibration_time_s += time.perf_counter() - t0

                # ---- 6. Commit or reject ----
                if result.committed:
                    t0 = time.perf_counter()
                    memory.save(delta, trust_weight=result.rho)
                    metrics.memory_time_s += time.perf_counter() - t0
                    metrics.n_deltas_committed += 1
                else:
                    metrics.n_deltas_rejected += 1

                if save_trajectory:
                    round_data.append({
                        "agent": agent_idx,
                        "rho": round(result.rho, 3),
                        "committed": result.committed,
                        "tokens": output.tokens_used,
                        "adversary": agent_idx in adversary_ids,
                    })

            if save_trajectory:
                task_traj["rounds"].append(round_data)

        # Evaluate
        answer = mas.get_final_answer()
        score = benchmark.evaluate_answer(task, answer)
        metrics.score += score

        if save_trajectory:
            task_traj["score"] = score
            trajectories.append(task_traj)

    # Normalize
    metrics.score = metrics.score / len(tasks) * 100

    out = asdict(metrics)
    if save_trajectory:
        out["trajectories"] = trajectories
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate
    if args.safeguard == "a_memguard" and is_graph_memory(args.memory):
        logger.error("A-MemGuard incompatible with graph memory."); sys.exit(1)
    if args.ablation and args.safeguard != "equimem":
        logger.error("--ablation only works with --safeguard equimem"); sys.exit(1)

    # Load
    logger.info(f"Config: {args.benchmark} | {args.mas} | {args.memory} | {args.safeguard}")
    benchmark = load_benchmark(args.benchmark)
    memory = load_memory(args.memory, args.backbone)
    safeguard = load_safeguard(args.safeguard, args.memory, args.ablation)
    mas = load_mas(args.mas, args.backbone, args.n_agents)
    attack = load_attack(args.attack, args.n_adversaries, args.n_agents)

    # Run
    all_results = []
    for run_id in range(args.n_runs):
        seed = args.seed if args.seed is not None else run_id
        logger.info(f"Run {run_id+1}/{args.n_runs} (seed={seed})")

        result = run_single(
            benchmark, mas, memory, safeguard, attack,
            args.max_rounds, seed, args.save_trajectory,
        )
        all_results.append(result)

        logger.info(
            f"  {result['score']:.1f}% | "
            f"tok={result['total_tokens']} | "
            f"commit={result['n_deltas_committed']}/{result['n_deltas_proposed']} | "
            f"t_agent={result['agent_time_s']:.1f}s "
            f"t_calib={result['calibration_time_s']:.1f}s"
        )

    # Aggregate
    scores = [r["score"] for r in all_results]
    mean_s, std_s = np.mean(scores), np.std(scores)
    mean_tok = np.mean([r["total_tokens"] for r in all_results])
    mean_commit = np.mean([
        r["n_deltas_committed"] / max(1, r["n_deltas_proposed"])
        for r in all_results
    ])

    logger.info(
        f"\n{'='*50}\n"
        f"  RESULT: {mean_s:.1f} ± {std_s:.1f}  |  "
        f"tokens={mean_tok:.0f}  |  commit={mean_commit:.1%}\n"
        f"{'='*50}"
    )

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    parts = [args.benchmark, args.mas, args.memory, args.safeguard]
    if args.ablation:
        parts.append(args.ablation)
    if args.attack:
        parts.append(f"{args.attack}_k{args.n_adversaries}")

    output_path = os.path.join(args.output_dir, "_".join(parts) + ".json")

    with open(output_path, "w") as f:
        json.dump({
            "config": vars(args),
            "mean": round(mean_s, 1),
            "std": round(std_s, 1),
            "mean_tokens": round(mean_tok, 0),
            "mean_commit_rate": round(mean_commit, 3),
            "timing": {
                "agent_s": round(np.mean([r["agent_time_s"] for r in all_results]), 1),
                "memory_s": round(np.mean([r["memory_time_s"] for r in all_results]), 1),
                "calibration_s": round(np.mean([r["calibration_time_s"] for r in all_results]), 1),
            },
            "runs": all_results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, default=str)

    logger.info(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
