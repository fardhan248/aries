import os, fitz, uuid, base64, asyncio
from cryptography.hazmat.primitives.ciphers.aead import AESCCM
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.gemini import gemini_embedding
from pathlib import Path
from string_utils.database_queries import DocumentQueries
import utils.contextmanager_utils as cm

load_dotenv()

ENC_KEY = base64.b64decode(os.getenv("KEY"))
queries = DocumentQueries()

aesccm = AESCCM(ENC_KEY)

async def encrypt(text: str) -> bytes:
    if not isinstance(text, str):
        text = str(text)
        
    nonce = os.urandom(13)
    ciphertext = aesccm.encrypt(nonce, text.encode(), None)
    
    return nonce + ciphertext
    
async def decrypt(data: bytes) -> str:
    nonce = data[:13]
    ciphertext = data[13:]
    plaintext = aesccm.decrypt(nonce, ciphertext, None)
    
    return plaintext.decode()
 
async def chunk_document(content_type, file_bytes):
    if content_type not in ("application/pdf", "application/epub+zip", "text/plain"):
        if file_bytes[:4] == b"%PDF":
            content_type = "application/pdf"
        elif file_bytes[:2] == b"PK":
            content_type = "application/epub+zip"
        else:
            content_type = "text/plain"
    
    filetype_map = {
        "application/pdf": "pdf",
        "application/epub+zip": "epub",
    }
    
    if content_type == "application/pdf" or content_type == "application/epub+zip":
        filetype = filetype_map.get(content_type, "pdf")
        doc = fitz.open(stream=file_bytes, filetype=filetype)
        len_doc = len(doc)
        
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
    else: # txt
        len_doc = 1
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1")
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
        
    return splitter.split_text(text), len_doc
 
## Upload user_document per chat, add TTL
async def save_chunks_session_to_db(pool, chunks, pages, filename, content_type, configurable, prompt):
    def to_uuid(value: str | uuid.UUID) -> uuid.UUID:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(value)
    
    user_id = to_uuid(configurable["user_id"])
    tenant_id = to_uuid(configurable["tenant_id"])
    thread_id = to_uuid(configurable["thread_id"])
    
    s_knowledge_id = uuid.uuid4()
    
    metadata = {
        "filename": filename,
        "content-type": content_type,
        "pages": pages,
        "len_chunks": len(chunks),
    }
    
    vector = await gemini_embedding.aembed_documents(chunks)

    encrypted_chunks = await asyncio.gather(
        *(encrypt(chunk) for chunk in chunks)
    )
    
    encrypt_metadata = await encrypt(metadata)

    records = [
        (uuid.uuid4(), s_knowledge_id, tenant_id, user_id, thread_id, chunk, vec)
        for chunk, vec in zip(encrypted_chunks, vector)
    ]
    
    chunk_ids = [str(r[0]) for r in records]
    
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                queries.INPUT_SESSION_KNOWLEDGES,
                s_knowledge_id, tenant_id, user_id, thread_id, encrypt_metadata
            )
            
            await conn.executemany(
                queries.INPUT_SESSION_KNOWLEDGE_CHUNKS,
                records
            )

    except Exception as e:
        print(e)
        return {"status": "error", "s_knowledge_id": 0, "user_id": 0, "content": str(e)}
        
    return {"status": "success", "s_knowledge_id": str(s_knowledge_id), "user_id": user_id, "metadata": metadata, "chunk_ids": chunk_ids}

async def put_new_knowledge_session(pool, f, config, prompt):
    filename = f.filename
    content_type = f.content_type
    file_bytes = await f.read()
    print("content_type", content_type, filename)
    configurable = config["configurable"]
    user_id = configurable["user_id"]
    thread_id = configurable["thread_id"]
    
    # Upload file to storage
    try:
        response = await (
            cm.supabase_client.storage.from_("knowledge_session").upload(
                file=file_bytes,
                path=f"{str(thread_id)}/{filename}",
                file_options={
                    "content-type": content_type,
                    "upsert": "false"
                }
            )
        )
        
        # Chunking document
        chunks, pages = await chunk_document(content_type, file_bytes)
        
        result = await save_chunks_session_to_db(pool, chunks, pages, filename, content_type, configurable, prompt)
        
        return result

    except Exception as e:
        print(e)
        return {"status": "double", "s_knowledge_id": 0, "user_id": 0, "content": str(e)}

## Tool: Put new knowledge (by tenant admin)
async def save_chunks_to_db(pool, chunks, pages, filename, content_type, tenant_id):
    if not isinstance(tenant_id, uuid.UUID):
        tenant_id = uuid.UUID(tenant_id)
    
    knowledge_id = uuid.uuid4()
    
    metadata = {
        "filename": filename,
        "content-type": content_type,
        "pages": pages,
        "len_chunks": len(chunks),
    }

    vector = await gemini_embedding.aembed_documents(chunks)
    
    encrypted_chunks = await asyncio.gather(
        *(encrypt(chunk) for chunk in chunks)
    )
    
    encrypted_metadata = await encrypt(metadata)

    records = [
        (uuid.uuid4(), knowledge_id, tenant_id, chunk, vec)
        for chunk, vec in zip(encrypted_chunks, vector)
    ] 
    
    chunk_ids = [str(r[0]) for r in records]
    
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                queries.INPUT_KNOWLEDGES,
                knowledge_id, tenant_id, encrypted_metadata
            )
            
            await conn.executemany(
                queries.INPUT_KNOWLEDGE_CHUNKS,
                records
            )

    except Exception as e:
        print(e)
        return {"status": "error", "knowledge_id": 0, "tenant_id": 0, "content": str(e)}
        
    return {"status": "success", "knowledge_id": str(knowledge_id), "tenant_id": tenant_id, "metadata": metadata, "chunk_ids": chunk_ids}

async def put_new_knowledge(db_pool, f, tenant_id):
    global pool
    pool = db_pool
    
    filename = f.filename
    content_type = f.content_type
    file_bytes = await f.read()
    print("content_type", content_type, filename)
    # Upload file to storage
    try:
        response = await (
            cm.supabase_client.storage.from_("knowledges").upload(
                file=file_bytes,
                path=f"{str(tenant_id)}/{filename}",
                file_options={
                    "content-type": content_type,
                    "upsert": "false"
                }
            )
        )
        
        # Chunking document
        chunks, pages = await chunk_document(content_type, file_bytes)
        
        result = await save_chunks_to_db(pool, chunks, pages, filename, content_type, tenant_id)
      
        return result
    
    except Exception as e:
        print(e)
        return {"status": "double", "s_knowledge_id": 0, "user_id": 0, "content": str(e)}