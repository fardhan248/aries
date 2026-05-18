from langchain_ollama import ChatOllama, OllamaEmbeddings

ollama_llm = ChatOllama(
    model="qwen_llm",
    base_url="http://ollama_llm:11434",
)

ollama_embedding = OllamaEmbeddings(
    model="qwen_embedding",
    base_url="http://ollama_embedding:11434",
    dimensions=1024,
)