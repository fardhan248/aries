#!/bin/bash

echo "Starting Ollama embedding server..."
ollama serve &

OLLAMA_PID=$!

echo "Waiting for Ollama..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

echo "Ollama is ready, creating the embedding model..."
ollama create qwen_embedding -f /model_files/Modelfile_embedding

wait $OLLAMA_PID
