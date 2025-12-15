# Multi-stage build for OptiVision
FROM continuumio/miniconda3:latest as builder

# Set working directory
WORKDIR /app

# Copy environment and setup files
COPY setup.sh .
COPY llama.cpp/ ./llama.cpp/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment with proper channels
RUN conda create -n vlm python=3.9 -y -c conda-forge
RUN echo "conda activate vlm" >> ~/.bashrc

# Build llama.cpp
SHELL ["/bin/bash", "--login", "-c"]
RUN conda activate vlm && cd llama.cpp && \
    mkdir -p build && \
    cd build && \
    cmake .. -DLLAMA_SERVER=ON -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --config Release

# Production stage
FROM continuumio/miniconda3:latest

WORKDIR /app

# Install system dependencies for runtime
RUN apt-get update && apt-get install -y \
    python3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy built artifacts
COPY --from=builder /app/llama.cpp/build ./llama.cpp/build

# Create conda environment in production stage
RUN conda create -n vlm python=3.9 -y -c conda-forge

# Copy application files
COPY *.html ./
COPY *.sh ./
COPY demo/ ./demo/
COPY videos/ ./videos/
COPY README.md LICENSE ./

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "🚀 Starting OptiVision..."\n\
\n\
# Activate conda environment\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate vlm\n\
\n\
# Start llama server in background\n\
cd llama.cpp/build\n\
model=$(cat ../selected_model.txt 2>/dev/null || echo "ggml-org/SmolVLM-500M-Instruct-GGUF")\n\
echo "📡 Starting backend with model: $model"\n\
./bin/llama-server -hf "$model" --host 0.0.0.0 --port 8080 &\n\
\n\
# Start frontend server\n\
cd /app\n\
echo "🌐 Starting frontend on port 3000"\n\
python3 -m http.server 3000 --bind 0.0.0.0\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose ports
EXPOSE 3000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/landing.html || exit 1

# Start application
CMD ["/app/start.sh"]
