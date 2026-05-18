#!/bin/bash

echo "Starting Ollama LLM server..."
ollama serve &

OLLAMA_PID=$!

echo "Waiting for Ollama..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

echo "Ollama is ready, creating the LLM model..."
ollama create qwen_llm -f /model_files/Modelfile

wait $OLLAMA_PID
