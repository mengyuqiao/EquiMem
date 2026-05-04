"""Main entry point for EquiMem experiments.

Usage:
    python tasks/run.py \
        --benchmark hotpotqa \
        --mas autogen \
        --memory gmemory \
        --safeguard equimem \
        --backbone qwen3:8b \
        --n_agents 6 \
        --n_runs 5
"""

import argparse
import json
import os
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Run EquiMem experiments")

    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["hotpotqa", "locomo", "alfworld", "webshop"],
                        help="Benchmark to evaluate on")
    parser.add_argument("--mas", type=str, required=True,
                        choices=["autogen", "macnet", "dylan"],
                        help="Multi-agent system framework")
    parser.add_argument("--memory", type=str, required=True,
                        choices=["none", "voyager", "memorybank", "arigraph", "gmemory"],
                        help="Memory architecture")
    parser.add_argument("--safeguard", type=str, required=True,
                        choices=["vanilla", "llm_audit", "ppl_filter", "a_memguard", "equimem"],
                        help="Safeguard method")
    parser.add_argument("--backbone", type=str, default="qwen3:8b",
                        help="LLM backbone model (Ollama model name)")
    parser.add_argument("--n_agents", type=int, default=6,
                        help="Number of agents")
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Number of runs with different seeds")
    parser.add_argument("--max_rounds", type=int, default=5,
                        help="Maximum debate rounds")
    parser.add_argument("--output_dir", type=str, default="results/",
                        help="Output directory")

    # Attack settings (for adversarial evaluation)
    parser.add_argument("--attack", type=str, default=None,
                        choices=["memory_poisoning", "audit_collusion"],
                        help="Attack type (None for benign)")
    parser.add_argument("--n_adversaries", type=int, default=0,
                        help="Number of adversarial agents")

    return parser.parse_args()


def load_benchmark(name: str):
    """Load benchmark dataset and evaluator."""
    if name == "hotpotqa":
        from .hotpotqa import HotpotQABenchmark
        return HotpotQABenchmark()
    elif name == "locomo":
        from .locomo import LoCoMoBenchmark
        return LoCoMoBenchmark()
    elif name == "alfworld":
        from .alfworld import ALFWorldBenchmark
        return ALFWorldBenchmark()
    elif name == "webshop":
        from .webshop import WebShopBenchmark
        return WebShopBenchmark()
    raise ValueError(f"Unknown benchmark: {name}")


def load_memory(name: str):
    """Load memory architecture."""
    if name == "none":
        return None
    elif name == "voyager":
        from memory.voyager import VoyagerMemory
        return VoyagerMemory()
    elif name == "memorybank":
        from memory.memorybank import MemoryBankMemory
        return MemoryBankMemory()
    elif name == "arigraph":
        from memory.arigraph import AriGraphMemory
        return AriGraphMemory()
    elif name == "gmemory":
        from memory.gmemory import GMemory
        return GMemory()
    raise ValueError(f"Unknown memory: {name}")


def load_safeguard(name: str, memory_type: str):
    """Load safeguard method."""
    if name == "vanilla":
        from baselines.vanilla import VanillaSafeguard
        return VanillaSafeguard()
    elif name == "llm_audit":
        from baselines.llm_audit import LLMAuditSafeguard
        return LLMAuditSafeguard()
    elif name == "ppl_filter":
        from baselines.ppl_filter import PPLFilterSafeguard
        return PPLFilterSafeguard()
    elif name == "a_memguard":
        if memory_type in ("arigraph", "gmemory"):
            raise ValueError("A-MemGuard is not compatible with graph memory")
        from baselines.a_memguard import AMemGuardSafeguard
        return AMemGuardSafeguard()
    elif name == "equimem":
        if memory_type in ("arigraph", "gmemory"):
            from equimem.calibration.graph import GraphCalibrator
            return GraphCalibrator()
        else:
            from equimem.calibration.embedding import EmbeddingCalibrator
            return EmbeddingCalibrator()
    raise ValueError(f"Unknown safeguard: {name}")


def load_mas(name: str, backbone: str, n_agents: int):
    """Load MAS framework."""
    if name == "autogen":
        from mas.autogen_adapter import AutoGenAdapter
        return AutoGenAdapter(backbone, n_agents)
    elif name == "macnet":
        from mas.macnet_adapter import MacNetAdapter
        return MacNetAdapter(backbone, n_agents)
    elif name == "dylan":
        from mas.dylan_adapter import DyLANAdapter
        return DyLANAdapter(backbone, n_agents)
    raise ValueError(f"Unknown MAS: {name}")


def main():
    args = parse_args()

    # Setup
    benchmark = load_benchmark(args.benchmark)
    memory = load_memory(args.memory)
    safeguard = load_safeguard(args.safeguard, args.memory)
    mas = load_mas(args.mas, args.backbone, args.n_agents)

    # Run
    all_results = []
    for run_id in range(args.n_runs):
        print(f"\n{'='*60}")
        print(f"Run {run_id+1}/{args.n_runs} | {args.benchmark} | "
              f"{args.mas} | {args.memory} | {args.safeguard}")
        print(f"{'='*60}")

        result = benchmark.evaluate(
            mas=mas,
            memory=memory,
            safeguard=safeguard,
            max_rounds=args.max_rounds,
            seed=run_id,
            attack=args.attack,
            n_adversaries=args.n_adversaries,
        )
        all_results.append(result)
        print(f"  Score: {result['score']:.1f}%")

    # Aggregate
    scores = [r["score"] for r in all_results]
    mean_score = sum(scores) / len(scores)
    std_score = (sum((s - mean_score)**2 for s in scores) / len(scores)) ** 0.5

    print(f"\n{'='*60}")
    print(f"Final: {mean_score:.1f} ± {std_score:.1f}")
    print(f"{'='*60}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"{args.benchmark}_{args.mas}_{args.memory}_{args.safeguard}.json"
    )
    with open(output_path, "w") as f:
        json.dump({
            "config": vars(args),
            "mean": mean_score,
            "std": std_score,
            "runs": all_results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
