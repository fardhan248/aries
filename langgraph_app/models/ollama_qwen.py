from langchain_ollama import ChatOllama, OllamaEmbeddings

ollama_llm = ChatOllama(
    model="Qwen3-0.6B-Q8_0",
    base_url="http://ollama_llm:11434",
)

ollama_embedding = OllamaEmbeddings(
    model="Qwen3-Embedding-0.6B-Q8_0",
    base_url="http://ollama_embedding:11434",
    dimensions=1024,
)