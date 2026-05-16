#!/bin/bash

echo "Starting Ollama embedding server..."
ollama serve &
sleep 5
echo "Ollama is ready, creating the embedding model..."

ollama create qwen_embedding -f model_files/Modelfile_embedding
tail -f /dev/null
#ollama run qwen_embedding
