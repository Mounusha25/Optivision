#!/bin/bash
set -e

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Load conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlm

# Go to build folder
cd llama.cpp/build

# Read model from saved file (or fallback)
model=$(cat ../selected_model.txt 2>/dev/null || echo "ggml-org/SmolVLM-500M-Instruct-GGUF")

echo "🚀 Starting llama-server with model: $model"
./bin/llama-server -hf "$model" --host 0.0.0.0 --port 8080
