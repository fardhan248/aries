#!/bin/bash

echo "Starting Ollama LLM server..."
ollama serve &
sleep 5
echo "Ollama is ready, creating the LLM model..."

ollama create qwen_llm -f model_files/Modelfile
tail -f /dev/null
#ollama run qwen_llm
