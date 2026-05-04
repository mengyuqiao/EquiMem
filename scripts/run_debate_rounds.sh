#!/bin/bash
# Reproduce debate rounds analysis (Appendix)
# MacNet, MemBank + G-Memory, LoCoMo, rounds 1/3/5

MEMORIES="memorybank gmemory"
SAFEGUARDS="vanilla llm_audit ppl_filter a_memguard equimem"
ROUNDS="1 3 5"

for mem in $MEMORIES; do
  for sg in $SAFEGUARDS; do
    # Skip A-MemGuard on graph memory
    if [[ "$sg" == "a_memguard" && "$mem" == "gmemory" ]]; then
      continue
    fi
    for r in $ROUNDS; do
      python tasks/run.py \
        --benchmark locomo --mas macnet --memory "$mem" \
        --safeguard "$sg" --max_rounds "$r" \
        --n_runs 5 --output_dir "results/debate_rounds/"
    done
  done
done
