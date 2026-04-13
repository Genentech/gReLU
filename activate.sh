#!/usr/bin/env bash
# Usage: source activate.sh

# This script should be sourced, not executed directly.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Please run this script using: source activate.sh"
    exit 1
fi

# Ensure conda is initialized (resolves CommandNotFoundError in non-interactive shells)
source /work/miniconda3/etc/profile.d/conda.sh

conda activate grelu_dev
echo "✅ Activated conda environment: grelu_dev"
