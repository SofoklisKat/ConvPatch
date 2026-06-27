#!/usr/bin/env bash
# CIFAR-100 ViT-Tiny sweep (all 5 patch-embed variants).
exec env DATASET=cifar100 "$(dirname "$0")/run_experiments.sh" "$@"
