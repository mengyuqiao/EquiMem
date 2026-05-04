#!/bin/bash
# Reproduce backbone comparison (Appendix)
# MacNet, MemBank + G-Memory, 3 backbones

BENCHMARKS="hotpotqa locomo alfworld webshop"
MEMORIES="memorybank gmemory"
BACKBONES="qwen3:8b deepseek-r1:7b llama3.1:8b"
SAFEGUARDS="vanilla llm_audit ppl_filter a_memguard equimem"

for backbone in $BACKBONES; do
  for bench in $BENCHMARKS; do
    for mem in $MEMORIES; do
      for sg in $SAFEGUARDS; do
        # Skip A-MemGuard on graph memory
        if [[ "$sg" == "a_memguard" && "$mem" == "gmemory" ]]; then
          continue
        fi
        python tasks/run.py \
          --benchmark "$bench" --mas macnet --memory "$mem" \
          --safeguard "$sg" --backbone "$backbone" \
          --n_runs 5 --output_dir "results/backbone/"
      done
    done
  done
done
