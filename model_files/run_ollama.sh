#!/bin/bash

echo "Starting Ollama LLM server..."
ollama serve &

echo "Ollama is ready, creating the LLM model..."

ollama create qwen_llm -f model_files/Modelfile
ollama run qwen_llm