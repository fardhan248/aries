# Chatbot-Aries

How to run:
1. Download Hugging Face Models
```
wget -P ./model_files/ https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
```

2. Run docker compose
```
docker compose -f docker-compose.yml up -d
```
