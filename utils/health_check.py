from models.gemini import gemini_instruct, gemini_thinking_reasoning, gemini_embedding
import utils.contextmanager_utils as cm

pool = None

async def llm_check(results):
    # gemini_instruct
    try:
        response = await gemini_instruct.ainvoke("ping")
        
        results["gemini_instruct"] = {"status": "success", "content": response.content}
    
    except Exception as e:
        results["gemini_instruct"] = {"status": "error", "content": str(e)}
        
    # gemini_thinking_reasoning
    try:
        response = await gemini_thinking_reasoning.ainvoke("ping")
        
        results["gemini_thinking_reasoning"] = {"status": "success", "content": response.content}
    
    except Exception as e:
        results["gemini_thinking_reasoning"] = {"status": "error", "content": str(e)}
        
    # gemini_embedding
    try:
        response = await gemini_embedding.aembed_query("ping")
        
        results["gemini_embedding"] = {"status": "success", "content": f"Vector length: {len(response)}"}
    
    except Exception as e:
        results["gemini_embedding"] = {"status": "error", "content": str(e)}

    return results
    
async def db_check(results):
    # asyncpg pool
    try:
        async with pool.acquire() as conn:
            response = await conn.fetchval("SELECT 1")
            results["asyncpg_pool"] = {"status": "success", "content": response == 1}
    
    except Exception as e:
        results["asyncpg_pool"] = {"status": "error", "content": str(e)}
    
    # supabase
    try:
        await (
            cm.supabase_client
            .table("checkpoint_migrations")
            .select("v")
            .limit(1)
            .execute()
        )
       
        results["supabase"] = {"status": "success", "content": True}
    
    except Exception as e:
        results["supabase"] = {"status": "error", "content": str(e)}
        
    return results
    
async def health_check(db_pool):
    global pool
    pool = db_pool
    
    results = {}
    
    # LLM
    results = await llm_check(results)  
        
    # Database
    results = await db_check(results)
    
    # Storage
    try:
        response = await cm.supabase_client.storage.list_buckets()
       
        results["supabase_storage"] = {"status": "success", "content": True}
    
    except Exception as e:
        results["supabase_storage"] = {"status": "error", "content": str(e)}

    return results