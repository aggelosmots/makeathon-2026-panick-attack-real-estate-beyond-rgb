#!/usr/bin/env sh
MODEL="${OLLAMA_MODEL:-qwen3:8b}"
HOST="${OLLAMA_HOST:-http://localhost:11434}"

echo "Pulling Ollama model: ${MODEL}"
docker compose exec ollama ollama pull "${MODEL}"
