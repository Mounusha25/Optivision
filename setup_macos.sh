#!/bin/bash

# Exit on error
set -e

# --------------------
# MODEL SELECTION
# --------------------
echo "📦 Available Models:"
models=(
    "ggml-org/SmolVLM-Instruct-GGUF"
    "ggml-org/SmolVLM-256M-Instruct-GGUF"
    "ggml-org/SmolVLM-500M-Instruct-GGUF"
    "ggml-org/SmolVLM2-2.2B-Instruct-GGUF"
    "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF"
    "ggml-org/SmolVLM2-500M-Video-Instruct-GGUF"
)

for i in "${!models[@]}"; do
    echo "$((i+1))) ${models[$i]}"
done

echo -n "➡️  Select model [1-${#models[@]}] (default 3): "
read -r selection
model="${models[${selection:-3}-1]}"

echo "✅ Selected model: $model"

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Save the selected model for future runs
echo "$model" > llama.cpp/selected_model.txt

# --------------------
# ACTIVATE CONDA ENV
# --------------------
echo "🔁 Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlm

# --------------------
# START SERVER
# --------------------
echo "🚀 Starting llama-server with model: $model"
cd llama.cpp/build
./bin/llama-server -hf "$model" --host 0.0.0.0 --port 8080
