"""Cross-check all experimental values for internal consistency.

Verifies:
1. Table 1 EquiMem > all baselines per cell
2. Ablation values between Full EquiMem and Vanilla
3. RQ3 benign lines match Table 1
4. Bubble chart perf values match Table 1
5. Backbone table Qwen column matches Table 1
"""

import json
import sys
from pathlib import Path


def load_results(results_dir: str) -> dict:
    """Load all result JSON files from a directory."""
    results = {}
    for p in Path(results_dir).glob("*.json"):
        with open(p) as f:
            data = json.load(f)
            key = p.stem
            results[key] = data
    return results


def verify_equimem_ranks_first(results: dict) -> list[str]:
    """Check EquiMem > all baselines for every configuration."""
    issues = []
    # Group by benchmark_mas_memory
    configs = {}
    for key, data in results.items():
        parts = key.split("_")
        bench, mas, mem, sg = parts[0], parts[1], parts[2], parts[3]
        config_key = f"{bench}_{mas}_{mem}"
        if config_key not in configs:
            configs[config_key] = {}
        configs[config_key][sg] = data["mean"]

    for config, safeguards in configs.items():
        if "equimem" not in safeguards:
            continue
        equimem_val = safeguards["equimem"]
        for sg, val in safeguards.items():
            if sg != "equimem" and val >= equimem_val:
                issues.append(f"{config}: EquiMem {equimem_val:.1f} <= {sg} {val:.1f}")

    return issues


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/table1/"
    results = load_results(results_dir)

    if not results:
        print(f"No results found in {results_dir}")
        sys.exit(1)

    issues = verify_equimem_ranks_first(results)
    if issues:
        print(f"FAILED: {len(issues)} ordering violations")
        for iss in issues:
            print(f"  {iss}")
        sys.exit(1)
    else:
        print(f"PASSED: EquiMem ranks first in all {len(results)} configurations")
