#!/bin/bash
# Reproduce Table 1: All configurations
# Estimated time: ~72 hours on 4x RTX 6000 Ada

BENCHMARKS="hotpotqa locomo alfworld webshop"
MAS_SYSTEMS="autogen macnet dylan"
MEMORIES="voyager memorybank arigraph gmemory"
SAFEGUARDS="vanilla llm_audit ppl_filter a_memguard equimem"
BACKBONE="qwen3:8b"
N_AGENTS=6
N_RUNS=5

echo "=========================================="
echo "EquiMem - Table 1 Reproduction"
echo "=========================================="
echo "Benchmarks: $BENCHMARKS"
echo "MAS: $MAS_SYSTEMS"
echo "Memories: $MEMORIES"
echo "Safeguards: $SAFEGUARDS"
echo "Backbone: $BACKBONE"
echo "Agents: $N_AGENTS, Runs: $N_RUNS"
echo "=========================================="

for bench in $BENCHMARKS; do
  for mas in $MAS_SYSTEMS; do
    for mem in $MEMORIES; do
      for sg in $SAFEGUARDS; do
        # Skip A-MemGuard on graph memory
        if [[ "$sg" == "a_memguard" && ("$mem" == "arigraph" || "$mem" == "gmemory") ]]; then
          echo "SKIP: $bench/$mas/$mem/$sg (A-MemGuard incompatible with graph)"
          continue
        fi

        echo "RUN: $bench | $mas | $mem | $sg"
        python tasks/run.py \
          --benchmark "$bench" \
          --mas "$mas" \
          --memory "$mem" \
          --safeguard "$sg" \
          --backbone "$BACKBONE" \
          --n_agents $N_AGENTS \
          --n_runs $N_RUNS \
          --output_dir "results/table1/"
      done
    done
  done
done

# Also run stateless baseline
for bench in $BENCHMARKS; do
  for mas in $MAS_SYSTEMS; do
    echo "RUN: $bench | $mas | none | vanilla (stateless)"
    python tasks/run.py \
      --benchmark "$bench" \
      --mas "$mas" \
      --memory "none" \
      --safeguard "vanilla" \
      --backbone "$BACKBONE" \
      --n_agents $N_AGENTS \
      --n_runs $N_RUNS \
      --output_dir "results/table1/"
  done
done

echo "=========================================="
echo "DONE. Results in results/table1/"
echo "=========================================="
