from utils.contextmanager_utils import supabase_client
from string_utils.database_queries import AddQueries
from utils.documents_utils import encrypt, decrypt
import os, uuid

pool = None
addqueries = AddQueries()

async def new_company(db_pool, tenant: str):
    global pool
    pool = db_pool
    
    tenant_id = uuid.uuid4()
    
    try:
        await pool.execute(
            addqueries.INSERT_NEW_COMPANY,
            tenant_id, tenant
        )
        
        return {"status": "success", "values": {"tenant_id": tenant_id, "name": tenant}}
        
    except Exception as e:
        print(e)
        return {"status": "Failed", "values": {}}
    
async def new_user(db_pool, tenant_id: str | uuid.UUID, user: str, role: str = "user"):
    global pool
    pool = db_pool
    
    if not isinstance(tenant_id, uuid.UUID):
        tenant_id = uuid.UUID(tenant_id)
    
    encrypted_user = await encrypt(user)
    user_id = uuid.uuid4()
    
    try:
        await pool.execute(
            addqueries.INSERT_NEW_USER,
            user_id, tenant_id, role, encrypted_user
        )
        
        return {"status": "success", "values": {"user_id": user_id, "tenant_id": tenant_id, "user": user, "role": role}}
    
    except Exception as e:
        print(e)
        return {"status": "Failed", "values": {}}   