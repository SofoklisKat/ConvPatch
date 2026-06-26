#!/usr/bin/env bash
# Alias for run_experiments.sh (CIFAR-10 sweep).
exec "$(dirname "$0")/run_experiments.sh" "$@"
