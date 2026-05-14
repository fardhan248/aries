import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# We recommend using the following set of sampling parameters for generation

# Thinking mode for general tasks: temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
# Thinking mode for precise coding tasks (e.g. WebDev): temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0
# Instruct (or non-thinking) mode for general tasks: temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
# Instruct (or non-thinking) mode for reasoning tasks: temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
# Please note that the support for sampling parameters varies according to inference frameworks.

# Ada 3 pilihan: Auto, thinking, fast

# LLM-Instruct (VL included) (or basic) 
llm_instruct = ChatOpenAI(
    base_url="http://localhost:8000/v1", # https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/main
    api_key="none",
    model="/models/Qwen3.5-4B-Q4_K_M" # Instruct (or non-thinking) mode for general tasks
)

# LLM-Thinking (VL included) (LLM in reasoning task)
llm_thinking = ChatOpenAI(
    base_url="http://localhost:8001/v1", # https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/main
    api_key="none",
    model="/models/Qwen3.5-9B-Q4_K_M" # Thinking mode for general tasks
)

# Embedding-VL
embedding_vl = OpenAIEmbeddings(
    base_url="http://localhost:8002/v1", # https://huggingface.co/DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF/tree/main
    api_key="none",
    model="/models/Qwen3-4B-Embedding-Q4_K_M"
)

# reasoning decision
llm_reasoning = ChatOpenAI(
    base_url="http://localhost:8003/v1", # https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/main
    api_key="none",
    model="/models/Qwen3.5-4B-Q4_K_M" # Instruct (or non-thinking) mode for reasoning tasks
)

# Coding (or use reasoning)
llm_coding = ChatOpenAI(
    base_url="http://localhost:8004/v1", # https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/main
    api_key="none",
    model="/models/Qwen3.5-4B-Q4_K_M" # Thinking mode for precise coding tasks
)

# Image/video generation?