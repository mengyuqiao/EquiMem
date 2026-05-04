#!/bin/bash
# Reproduce RQ2: Adversarial robustness
# MacNet, MemBank + G-Memory, HotpotQA + ALFWorld

BENCHMARKS="hotpotqa alfworld"
MEMORIES="memorybank gmemory"
ATTACKS="memory_poisoning audit_collusion"
K_VALUES="1 2 3"
SAFEGUARDS="equimem llm_audit ppl_filter"

for bench in $BENCHMARKS; do
  for mem in $MEMORIES; do
    for attack in $ATTACKS; do
      for k in $K_VALUES; do
        for sg in $SAFEGUARDS; do
          echo "RUN: $bench | $mem | $sg | $attack | k=$k"
          python tasks/run.py \
            --benchmark "$bench" --mas macnet --memory "$mem" \
            --safeguard "$sg" --n_runs 5 \
            --attack "$attack" --n_adversaries "$k" \
            --output_dir "results/adversarial/"
        done
      done
    done
  done
done
