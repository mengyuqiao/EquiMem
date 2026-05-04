#!/bin/bash
# Reproduce ablation study (Appendix)
# MemBank + G-Memory, all MAS, all benchmarks

BENCHMARKS="hotpotqa locomo alfworld webshop"
MAS_SYSTEMS="autogen macnet dylan"
MEMORIES="memorybank gmemory"
ABLATIONS="wo_detect wo_align wo_reweight"

for bench in $BENCHMARKS; do
  for mas in $MAS_SYSTEMS; do
    for mem in $MEMORIES; do
      # Full EquiMem (reference)
      python tasks/run.py \
        --benchmark "$bench" --mas "$mas" --memory "$mem" \
        --safeguard equimem --n_runs 5 \
        --output_dir "results/ablation/"

      # Each ablation
      for abl in $ABLATIONS; do
        python tasks/run.py \
          --benchmark "$bench" --mas "$mas" --memory "$mem" \
          --safeguard equimem --n_runs 5 \
          --ablation_config "configs/ablations/${abl}.yaml" \
          --output_dir "results/ablation/"
      done
    done
  done
done
