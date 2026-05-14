#!/bin/bash

echo "Starting Ollama server..."
ollama serve &

echo "Ollama is ready, creating the model..."

ollama create qwen_embedding -f model_files/Modelfile_embedding
ollama run qwen_embedding