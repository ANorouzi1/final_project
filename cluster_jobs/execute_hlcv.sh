#!/usr/bin/env bash
set -euo pipefail

# Edit these two paths after copying the project to the cluster.
PROJECT_DIR="/home/hlcv_teamxxx/HLCV_final_project"
CONDA_PYTHON_BINARY_PATH="/home/hlcv_teamxxx/miniconda3/envs/hlcv/bin/python"

cd "$PROJECT_DIR"
"$CONDA_PYTHON_BINARY_PATH" "$@"
