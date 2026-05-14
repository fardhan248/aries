#!/bin/bash

echo "Starting Ollama embedding server..."
ollama serve &

echo "Ollama is ready, creating the embedding model..."

ollama create qwen_embedding -f model_files/Modelfile_embedding
ollama run qwen_embedding