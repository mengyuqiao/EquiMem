# EquiMem: Zero-Trust Memory Calibration for Multi-Agent LLM Debate

> **Anonymous repository for NeurIPS 2026 submission.**  
> This codebase implements EquiMem, an inference-time memory calibration mechanism for shared-memory multi-agent debate systems. EquiMem adds zero token overhead and requires no LLM calls for calibration.

---

## Overview

Multi-agent debate (MAD) systems augment LLM agents with shared memory to support cross-trial learning. However, shared memory introduces a critical vulnerability: once an erroneous entry is committed, it corrupts all downstream retrievals. Existing safeguards (LLM Audit, PPL Filter, A-MemGuard) score each entry in isolation and cannot detect entries that are locally plausible but globally inconsistent.

**EquiMem** formalizes memory updating as a *zero-trust game* and implements an LLM-free calibration mechanism Φ that:
- Intercepts agents' existing retrieval activity as evidence probes (zero additional tokens)
- Scores each proposed update against the full memory state (not in isolation)
- Works on both embedding-based and graph-based memory architectures
- Degrades gracefully under adversarial conditions (up to 50% adversarial agents)

## Repository Structure

```
EquiMem/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── default.yaml              # Default experiment config
│   ├── benchmarks/
│   │   ├── hotpotqa.yaml
│   │   ├── locomo.yaml
│   │   ├── alfworld.yaml
│   │   └── webshop.yaml
│   └── ablations/
│       ├── wo_detect.yaml
│       ├── wo_align.yaml
│       └── wo_reweight.yaml
│
├── EquiMem/                        # Core EquiMem implementation
│   ├── __init__.py
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract calibration interface
│   │   ├── embedding.py          # ρ_detect + ρ_align for embedding memory
│   │   ├── graph.py              # s_local + s_path + ρ_align for graph memory
│   │   ├── credibility.py        # Bidirectional credibility weight w_i
│   │   └── threshold.py          # Adaptive threshold ρ* (running median)
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── trust_discount.py     # Trust-discounted retrieval (§4.3)
│   │   └── ann_index.py          # ANN index wrapper
│   └── utils/
│       ├── rbo.py                # Rank-biased overlap distance
│       ├── graph_bfs.py          # Bidirectional BFS for s_path
│       └── compat.py             # Relation compatibility estimation
│
├── memory/                        # Memory architecture adapters
│   ├── __init__.py
│   ├── base.py                   # MemoryBase interface (save/search)
│   ├── voyager.py                # Voyager adapter
│   ├── memorybank.py             # MemoryBank adapter
│   ├── arigraph.py               # AriGraph adapter (graph + episodic)
│   └── gmemory.py                # G-Memory adapter (hierarchical graph)
│
├── mas/                           # Multi-agent system integrations
│   ├── __init__.py
│   ├── base.py                   # MAS base interface
│   ├── autogen_adapter.py        # AutoGen integration
│   ├── macnet_adapter.py         # MacNet integration
│   └── dylan_adapter.py          # DyLAN integration
│
├── baselines/                     # Safeguard baselines
│   ├── __init__.py
│   ├── vanilla.py                # No defence (passthrough)
│   ├── llm_audit.py              # LLM-based audit per delta
│   ├── ppl_filter.py             # Perplexity-based filtering
│   └── a_memguard.py             # A-MemGuard (embedding only)
│
├── attacks/                       # Adversarial attack implementations
│   ├── __init__.py
│   ├── memory_poisoning.py       # Memory Poisoning attack (RQ2)
│   └── audit_collusion.py        # Audit Collusion attack (RQ2)
│
├── tasks/                         # Benchmark task runners
│   ├── __init__.py
│   ├── run.py                    # Main entry point
│   ├── hotpotqa.py               # HotpotQA evaluation
│   ├── locomo.py                 # LoCoMo evaluation
│   ├── alfworld.py               # ALFWorld evaluation
│   └── webshop.py                # WebShop evaluation
│
├── scripts/                       # Reproduction scripts
│   ├── run_table1.sh             # Reproduce Table 1 (all configs)
│   ├── run_ablation.sh           # Reproduce ablation study
│   ├── run_adversarial.sh        # Reproduce RQ2 adversarial
│   ├── run_backbone.sh           # Reproduce backbone comparison
│   └── run_debate_rounds.sh      # Reproduce debate rounds analysis
│
├── analysis/                      # Plotting and analysis
│   ├── plot_cost_analysis.py     # Bubble chart (Figure 3)
│   ├── plot_rq3_EquiMem.py        # Adversarial bar chart (Figure 2)
│   ├── plot_rq3_baselines.py     # Baseline adversarial (Appendix)
│   ├── plot_debate_rounds.py     # Debate rounds (Appendix)
│   └── verify_alignment.py       # Cross-check table values
│
└── tests/                         # Unit tests
    ├── test_calibration.py
    ├── test_credibility.py
    ├── test_rbo.py
    ├── test_graph_bfs.py
    └── test_trust_discount.py
```

## Quick Start

### Installation

```bash
git clone https://github.com/anonymous-EquiMem/EquiMem.git
cd EquiMem
pip install -e .
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- [Ollama](https://ollama.ai/) with `qwen3:8b` pulled
- See `requirements.txt` for full dependencies

```bash
# Pull the default backbone
ollama pull qwen3:8b
```

### Run a Single Experiment

```bash
python tasks/run.py \
  --benchmark hotpotqa \
  --mas autogen \
  --memory gmemory \
  --safeguard EquiMem \
  --backbone qwen3:8b \
  --n_agents 6 \
  --n_runs 5
```

### Reproduce Table 1

```bash
bash scripts/run_table1.sh
```

This runs all 3 MAS × 4 memory × 5 safeguard × 4 benchmark configurations (240 experiments, each repeated 5 times). Estimated time: ~72 hours on 4× RTX 6000 Ada.

### Reproduce Specific RQs

```bash
# RQ2: Adversarial robustness (MacNet, MemBank + G-Memory, HQA + ALF)
bash scripts/run_adversarial.sh

# RQ3: Token cost analysis (AutoGen, MemBank + G-Memory, HQA + ALF)
# (Token counts are logged automatically during run_table1.sh)

# Ablation study (all MAS, MemBank + G-Memory)
bash scripts/run_ablation.sh

# Alternative backbones (MacNet, MemBank + G-Memory)
bash scripts/run_backbone.sh

# Debate rounds (MacNet, MemBank + G-Memory, LoCoMo)
bash scripts/run_debate_rounds.sh
```

## Key Implementation Details

### Calibration Module (`EquiMem/calibration/`)

The calibration mechanism Φ is implemented as a standalone module with **no LLM dependency**. It exposes a single interface:

```python
from EquiMem.calibration import EmbeddingCalibrator, GraphCalibrator

# For embedding memory
calibrator = EmbeddingCalibrator()
rho = calibrator.score(delta_t, memory_state, auditor_probes)
# rho = sqrt(rho_detect * rho_align)

# For graph memory
calibrator = GraphCalibrator()
rho = calibrator.score(delta_edges, graph, auditor_paths)
# rho = sqrt(rho_detect * rho_align)
# where rho_detect = alpha * s_local + (1-alpha) * s_path

# Commit decision
if rho >= calibrator.threshold:  # adaptive running median
    memory.commit(delta_t, trust_weight=rho)
```

### Memory Adapters (`memory/`)

Each memory adapter implements the `MemoryBase` interface:

```python
class MemoryBase:
    def save(self, content: str, trust_weight: float = 1.0) -> None: ...
    def search(self, query: str, top_k: int = 5) -> list[str]: ...
    def get_state(self) -> MemoryState: ...
```

### Safeguard Interface (`baselines/`)

All safeguards (including EquiMem) implement:

```python
class SafeguardBase:
    def evaluate(self, delta, memory_state, **kwargs) -> float:
        """Return a score in [0, 1]. Higher = more trustworthy."""
        ...
```

## Configuration

All experiments are configured via YAML files. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_agents` | 6 | Number of agents in MAD |
| `top_k` | 5 | Retrieval top-k |
| `embedding_dim` | 384 | MiniLM embedding dimension |
| `rho_star_method` | `running_median` | Adaptive threshold |
| `L_max` | `log(|V|)` | BFS depth for graph |
| `delta_t` | `invalid_probe_ratio` | Credibility update rate |

All thresholds are **adaptive** (estimated from data). No manual tuning required.

## Benchmarks

| Benchmark | Type | Metric | Source |
|-----------|------|--------|--------|
| HotpotQA | Reasoning | Accuracy | [yang2018hotpotqa](https://hotpotqa.github.io/) |
| LoCoMo | Reasoning | F1 | [maharana2024evaluating](https://github.com/snap-research/locomo) |
| ALFWorld | Action | Success Rate | [shridhar2020alfworld](https://github.com/alfworld/alfworld) |
| WebShop | Action | Reward | [yao2022webshop](https://webshop-pnlp.github.io/) |

## Hardware

Experiments were conducted on:
- **GPU**: NVIDIA RTX 6000 Ada Generation (49 GB VRAM) × 4
- **LLM serving**: Ollama
- **Calibration**: CPU-only (no GPU required for Φ)

## Citation

```
Anonymous submission to NeurIPS 2026.
```

## License

MIT License
