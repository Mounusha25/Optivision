#!/bin/bash
# Railway startup script

set -e

echo "🚀 Starting OptiVision on Railway..."

# Install system dependencies if needed
if ! command -v cmake &> /dev/null; then
    echo "Installing build dependencies..."
    apt-get update && apt-get install -y build-essential cmake git
fi

# Setup conda environment
if [ ! -d "/opt/conda/envs/vlm" ]; then
    echo "Creating conda environment..."
    conda create -n vlm python=3.9 -y
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlm

# Build llama.cpp if not already built
if [ ! -f "llama.cpp/build/bin/llama-server" ]; then
    echo "Building llama.cpp..."
    cd llama.cpp
    mkdir -p build
    cd build
    cmake .. -DLLAMA_SERVER=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build . --config Release
    cd ../..
fi

# Start the backend server in background
echo "Starting backend server..."
cd llama.cpp/build
model=$(cat ../selected_model.txt 2>/dev/null || echo "ggml-org/SmolVLM-500M-Instruct-GGUF")
./bin/llama-server -hf "$model" --host 0.0.0.0 --port ${PORT:-8080} &

# Start frontend server
cd /app
echo "Starting frontend server on port ${PORT:-3000}..."
python3 -m http.server ${PORT:-3000} --bind 0.0.0.0
