# EquiMem: Zero-Trust Memory Calibration for Multi-Agent LLM Debate

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
├── vendor/GMemory/          # G-Memory codebase (git submodule), provides
│                            #   memory implementations, MAS frameworks, and
│                            #   ALFWorld environment
│
├── equimem/                 # Core calibration mechanism Φ (LLM-free)
│   ├── calibration/         #   Embedding and graph calibration scoring
│   ├── retrieval/           #   Trust-discounted retrieval (§4.3)
│   └── utils/               #   RBO distance, BFS, relation compatibility
│
├── adapters/                # Thin routing layer connecting vendor code,
│                            #   memory adapters, and safeguards to run.py
│
├── memory/                  # Memory architecture adapters wrapping vendor
│                            #   code into EquiMem's EmbeddingMemory /
│                            #   GraphMemory interfaces (+ new AriGraph)
│
├── mas/                     # MAS framework adapters wrapping vendor code
│                            #   with calibration injection and probe
│                            #   interception
│
├── baselines/               # Safeguard baselines: Vanilla, LLM Audit,
│                            #   PPL Filter, A-MemGuard
│
├── attacks/                 # Adversarial attacks for RQ2: Memory Poisoning
│                            #   and Audit Collusion
│
├── benchmarks/              # Benchmark loaders and evaluators: HotpotQA,
│                            #   LoCoMo, ALFWorld, WebShop
│
├── tasks/                   # Experiment entry point (run.py)
├── scripts/                 # Shell scripts to reproduce each table/figure
├── configs/                 # YAML configs for benchmarks and ablations
├── analysis/                # Result verification and plotting utilities
└── tests/                   # Unit tests for calibration components
```

## Setup

### Prerequisites

- Python ≥ 3.10
- PyTorch ≥ 2.1
- [Ollama](https://ollama.ai/) with `qwen3:8b` pulled

### Installation

```bash
git clone https://github.com/anonymous-equimem/EquiMem.git
cd EquiMem

# Initialize the G-Memory submodule
git submodule update --init --recursive

# Install EquiMem
pip install -e .
```

### Vendor Dependencies

This project builds on [G-Memory](https://github.com/bingreeky/GMemory) (Zhang et al., 2025) for memory architecture implementations and MAS framework integrations. G-Memory is included as a git submodule under `vendor/GMemory/` and provides:

- **Memory implementations**: Voyager, MemoryBank, G-Memory (our adapters in `memory/` wrap these)
- **MAS frameworks**: AutoGen, MacNet, DyLAN (our adapters in `mas/` wrap these)
- **ALFWorld environment**: Task runner and data for embodied evaluation

EquiMem adds the following components that are **not** in G-Memory:
- Calibration mechanism Φ (`equimem/calibration/`)
- AriGraph memory adapter (`memory/arigraph.py`)
- All safeguard baselines (`baselines/`)
- Adversarial attacks (`attacks/`)
- HotpotQA, LoCoMo, and WebShop benchmark runners (`benchmarks/`)

## Quick Start

### Run a Single Experiment

```bash
python tasks/run.py \
  --benchmark hotpotqa \
  --mas macnet \
  --memory gmemory \
  --safeguard equimem \
  --backbone qwen3:8b \
  --n_agents 6 \
  --n_runs 5
```

### Reproduce Main Results

```bash
# Table 1: All configurations (3 MAS × 4 memory × 5 safeguard × 4 benchmark)
# Estimated: ~72 hours on 4× RTX 6000 Ada
bash scripts/run_table1.sh

# RQ2: Adversarial robustness
bash scripts/run_adversarial.sh

# Ablation study
bash scripts/run_ablation.sh

# Alternative backbones (DeepSeek-R1-Distill-7B, Llama-3.1-8B)
bash scripts/run_backbone.sh

# Effect of debate rounds
bash scripts/run_debate_rounds.sh
```

### Verify Result Consistency

```bash
python analysis/verify_alignment.py results/table1/
```

## Architecture

### How EquiMem Integrates with G-Memory

```
┌─────────────────────────────────────────────────────┐
│  tasks/run.py  (experiment entry point)             │
│                                                     │
│  for each task:                                     │
│    for each round:                                  │
│      for each agent:                                │
│        1. memory.search(query)          ← vendor    │
│        2. mas.run_agent_turn(...)       ← vendor    │
│        3. [attack.modify_delta(...)]                 │
│        4. intercept auditor probes      ← EquiMem   │
│        5. safeguard.score(delta, M_t)   ← EquiMem   │
│        6. if committed: memory.save()   ← vendor    │
└─────────────────────────────────────────────────────┘

Steps 1, 2, 6 → delegate to G-Memory vendor code via adapters
Steps 4, 5    → EquiMem calibration (LLM-free, zero tokens)
Step  3       → adversarial evaluation only
```

### Calibration Module (`equimem/calibration/`)

The calibration mechanism Φ is a standalone Python module with **no LLM dependency**:

```python
from equimem.calibration import EmbeddingCalibrator, GraphCalibrator

# Embedding memory (Voyager, MemoryBank)
calibrator = EmbeddingCalibrator()
result = calibrator.score(delta, memory_state, auditor_probes)
# result.rho = sqrt(rho_detect * rho_align)

# Graph memory (AriGraph, G-Memory)
calibrator = GraphCalibrator()
result = calibrator.score(delta_edges, graph, auditor_paths)
# result.rho = sqrt(rho_detect * rho_align)
# where rho_detect = alpha * s_local + (1 - alpha) * s_path

if result.committed:  # rho >= adaptive threshold ρ*
    memory.save(delta, trust_weight=result.rho)
```

### Memory Adapters (`memory/`)

Each adapter wraps G-Memory's vendor code into a common interface:

```python
class EmbeddingMemory(MemoryBase):
    def save(self, content, trust_weight=1.0): ...
    def search(self, query, top_k=5): ...
    def similarities(self, vector): ...       # for ρ_detect
    def search_with_delta(self, query, delta, top_k): ...  # for ρ_align

class GraphMemory(MemoryBase):
    def save(self, content, trust_weight=1.0): ...
    def search(self, query, top_k=5): ...
    def get_neighbors(self, node): ...        # for s_local
    def get_compatible_relations(self, r): ... # for s_path
```

### Safeguard Interface (`baselines/`)

All safeguards (including EquiMem) share the same interface:

```python
class SafeguardBase:
    def score(self, delta, memory_state, auditor_signals) -> CalibrationResult:
        """Return CalibrationResult with rho ∈ [0, 1] and commit decision."""
```

## Configuration

All thresholds are **adaptive** (estimated from data). No manual tuning required.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_agents` | 6 | Number of agents in MAD |
| `backbone` | `qwen3:8b` | LLM backbone via Ollama |
| `top_k` | 5 | Retrieval depth |
| `embedding_dim` | 384 | MiniLM embedding dimension |
| `rho_star` | running median | Adaptive commit threshold |
| `L_max` | ⌈log \|V\|⌉ | BFS depth for graph calibration |
| `delta_t` | invalid probe ratio | Credibility update rate |

## Benchmarks

| Benchmark | Type | Metric | Source |
|-----------|------|--------|--------|
| HotpotQA | Reasoning | Accuracy | [yang2018hotpotqa](https://hotpotqa.github.io/) |
| LoCoMo | Reasoning | F1 | [maharana2024evaluating](https://github.com/snap-research/locomo) |
| ALFWorld | Action | Success Rate | [shridhar2020alfworld](https://github.com/alfworld/alfworld) |
| WebShop | Action | Reward | [yao2022webshop](https://webshop-pnlp.github.io/) |

## Hardware

- **GPU**: NVIDIA RTX 6000 Ada Generation (49 GB VRAM) × 4
- **LLM serving**: Ollama (local deployment)
- **Calibration**: CPU-only (no GPU required for Φ)
- **Driver**: NVIDIA 580.95.05, CUDA 13.0

## Acknowledgements

This codebase builds on [G-Memory](https://github.com/bingreeky/GMemory) (Zhang et al., 2025) for memory architecture implementations and MAS framework integrations. We thank the G-Memory authors for releasing their code under the MIT licence.


## License

MIT License