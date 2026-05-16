from utils.contextmanager_utils import supabase_client
from string_utils.database_queries import AddQueries
from utils.documents_utils import encrypt, decrypt
from string_utils.prompts import PromptTitle
from utils.documents_utils import encrypt, decrypt
from src.chat_completion import streaming, chat_workflow
from models.gemini import gemini
import os, uuid

pool = None
addqueries = AddQueries()
prompttitle = PromptTitle()

async def new_company(db_pool, tenant: str):
    global pool
    pool = db_pool
    
    tenant_id = str(uuid.uuid4())
    
    try:
        await pool.execute(
            addqueries.INSERT_NEW_COMPANY,
            tenant_id, tenant
        )
        
        return {"status": "success", "values": {"tenant_id": tenant_id, "name": tenant}}
        
    except Exception as e:
        print(e)
        return {"status": "Failed", "values": {}}
    
async def new_user(db_pool, tenant_id: str, user: str, role: str = "user"):
    global pool
    pool = db_pool
    
    encrypted_user = await encrypt(user)
    user_id = str(uuid.uuid4())
    
    try:
        await pool.execute(
            addqueries.INSERT_NEW_USER,
            user_id, tenant_id, role, encrypted_user
        )
        
        return {"status": "success", "values": {"user_id": user_id, "tenant_id": tenant_id, "user": user, "role": role}}
    
    except Exception as e:
        print(e)
        return {"status": "Failed", "values": {}}   
        
async def new_chat(db_pool, input_data: dict):
    global pool
    pool = db_pool
    
    user_id = input_data.user_id
    tenant_id = input_data.tenant_id
    input_prompt = input_data.input_prompt
    thread_id = str(uuid.uuid4())

    try:    
        async with pool.acquire() as conn:
            SESSION_TITLE_SYSTEM_QUERY = prompttitle.SESSION_TITLE_SYSTEM_QUERY.format_map({
                "input_prompt": input_prompt,
            })
            
            title = await gemini.ainvoke(SESSION_TITLE_SYSTEM_QUERY)
            encrypted_title = await encrypt(title)
            
            await conn.execute(
                addqueries.INSERT_SESSION_TITLE,
                thread_id, tenant_id, user_id, encrypted_title
            )                 
            
            return {"status": "success", "content": f"Session created {title}", "title": title, "thread_id": thread_id}
    
    except Exception as e:
        print(e)
        return {"status": "error", "content": str(e), "title": None, "thread_id": None}