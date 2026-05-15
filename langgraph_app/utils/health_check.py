from models.ollama_qwen import ollama_llm, ollama_embedding
import httpx

pool = None

async def llm_check(results):
    # ollama_llm (qwen)
    try:
        response = await ollama_llm.ainvoke("ping")
        
        results["ollama_llm"] = {"status": "success", "content": response.content}
    
    except Exception as e:
        results["ollama_llm"] = {"status": "error", "content": str(e)}
        
    # ollama_embedding (qwen)
    try:
        response = await ollama_embedding.aembed_query("ping")
        
        results["ollama_embedding"] = {"status": "success", "content": f"Vector length: {len(response)}"}
    
    except Exception as e:
        results["ollama_embedding"] = {"status": "error", "content": str(e)}

    return results
    
async def db_check(results):
    # asyncpg pool
    try:
        async with pool.acquire() as conn:
            response = await conn.fetchval("SELECT 1")
            results["asyncpg_pool"] = {"status": "success", "content": response == 1}
    
    except Exception as e:
        results["asyncpg_pool"] = {"status": "error", "content": str(e)}
    
    # ChromaDB
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://chromadb:8000/api/v1/heartbeat"
            )
            
            results["chromadb"] = {"status": "success", "content": response.text}
    
    except Exception as e:
        results["chromadb"] = {"status": "error", "content": str(e)}
        
    return results
    
async def health_check(db_pool):
    global pool
    pool = db_pool
    
    results = {}
    
    # LLM
    results = await llm_check(results)  
        
    # Database
    results = await db_check(results)

    return results