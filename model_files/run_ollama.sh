#!/bin/bash

echo "Starting Ollama server..."
ollama serve &

echo "Ollama is ready, creating the model..."

ollama create qwen_llm -f model_files/Modelfile
ollama run qwen_llm