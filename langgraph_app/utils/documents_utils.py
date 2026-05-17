import os, fitz, uuid, base64, asyncio
from cryptography.hazmat.primitives.ciphers.aead import AESCCM
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from models.ollama_qwen import ollama_embedding
from models.gemini import gemini_embedding
from langchain_chroma import Chroma
from langchain_core.documents import Document
from datetime import datetime, timedelta
import utils.contextmanager_utils as cm

ollama_embedding = gemini_embedding

load_dotenv()

ENC_KEY = base64.b64decode(os.getenv("KEY"))
aesccm = AESCCM(ENC_KEY)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)


async def get_vector_store_chroma(collection: str):
    return Chroma(
        client=cm.chroma,
        collection_name=collection,
        embedding_function=ollama_embedding,
        collection_metadata={"hnsw:space": "cosine"},
    )
    
async def get_vector_store_retriever(chroma_vector_store, search_filter: dict = None):
    search_kwargs = {"k": 5}
    
    if search_filter:
        search_kwargs["filter"] = search_filter
    
    return chroma_vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )
     
async def str_to_datetime(date: str):
    try:
        dt = datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")
    except Exception:
        dt = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    return dt

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
 
async def chunk_document(filename, content_type, file_bytes, configurable):
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
            
    chunks = splitter.split_text(text)
            
    tenant_id = configurable["tenant_id"]
    user_id = configurable.get("user_id", None)
    thread_id = configurable.get("thread_id", None)
    knowledge_id = str(uuid.uuid4())
    
    metadatas = [
        {
            "filename": filename,
            "content_type": content_type,
            "len_pages": len_doc,
            "number_chunks": len(chunks),
            "len_char": len(chunk),
            "tenant_id": tenant_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "knowledge_id": knowledge_id,
            "chunk_id": str(uuid.uuid4()),
            "created_at": str(datetime.now()),
        }
        for chunk in chunks
    ]
        
    return chunks, metadatas
 
## Upload user_document per chat, add TTL
async def save_chunks_session_to_db(chunks, metadatas):
    for i in range(len(metadatas)):
        created_at = await str_to_datetime(metadatas[i]["created_at"])
        metadatas[i]["expired_at"] = str(created_at + timedelta(days=7))
    
    thread_id = metadatas[0]["thread_id"]
    s_knowledge_id = metadatas[0]["knowledge_id"]
    
    documents = [
        Document(
            page_content=chunks[i],
            metadata=metadatas[i],
            id=metadatas[i]["chunk_id"],
        )
        for i in range(len(chunks))
    ]
    
    uuids = [x["chunk_id"] for x in metadatas]
    
    try:
        vector_store = await get_vector_store_chroma(f"thread_{thread_id.replace('-', '_')}")
        
        await vector_store.aadd_documents(documents=documents, ids=uuids)

    except Exception as e:
        print(e)
        return {"status": "error", "s_knowledge_id": 0, "thread_id": thread_id, "content": str(e)}
        
    return {"status": "success", "s_knowledge_id": s_knowledge_id, "thread_id": thread_id, "chunk_ids": uuids, "metadata": metadatas}

async def put_new_knowledge_session(f, config):    
    filename = f.filename
    content_type = f.content_type
    file_bytes = await f.read()
    
    configurable = config["configurable"]
    
    # Upload file to storage
    try:        
        # Chunking document
        chunks, metadatas = await chunk_document(filename, content_type, file_bytes, configurable)
        
        result = await save_chunks_session_to_db(chunks, metadatas)
        
        return result

    except Exception as e:
        print(e)
        return {"status": "error", "s_knowledge_id": 0, "user_id": 0, "content": str(e)}

## Tool: Put new knowledge (by tenant admin)
async def save_chunks_to_db(chunks, metadatas):
    tenant_id = metadatas[0]["tenant_id"]
    knowledge_id = metadatas[0]["knowledge_id"]
    
    documents = [
        Document(
            page_content=chunks[i],
            metadata=metadatas[i],
            id=metadatas[i]["chunk_id"],
        )
        for i in range(len(chunks))
    ]
    
    uuids = [x["chunk_id"] for x in metadatas]
    
    try:
        vector_store = await get_vector_store_chroma(f"tenant_{tenant_id.replace('-', '_')}")
        
        await vector_store.aadd_documents(documents=documents, ids=uuids)

    except Exception as e:
        print(e)
        return {"status": "error", "knowledge_id": 0, "tenant_id": tenant_id, "content": str(e)}
        
    return {"status": "success", "knowledge_id": knowledge_id, "tenant_id": tenant_id, "chunk_ids": uuids, "metadata": metadatas}
    
async def put_new_knowledge(f, tenant_id):    
    filename = f.filename
    content_type = f.content_type
    file_bytes = await f.read()
    
    try:
        # Chunking document
        chunks, metadatas = await chunk_document(filename, content_type, file_bytes, {"tenant_id": tenant_id})
        
        result = await save_chunks_to_db(chunks, metadatas)
      
        return result
    
    except Exception as e:
        print(e)
        return {"status": "error", "knowledge_id": 0, "tenant_id": 0, "content": str(e)}
        
async def delete_knowledge(tenant_id: str, knowledge_id: str):
    try:
        vector_store = await get_vector_store_chroma(f"tenant_{tenant_id.replace('-', '_')}")
        
        collection = vector_store._collection
        collection_name = collection.name
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: collection.delete(where={"knowledge_id": knowledge_id})
        )
        
        knowledge_ids = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: collection.get(where={"tenant_id": tenant_id}, include=[])
        )

        if len(knowledge_ids["ids"]) == 0:
            client = vector_store._client
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.delete_collection(collection_name)
            )
        
        return {"status": "success", "content": f"Delete knowledge_id {knowledge_id} success."}
    
    except Exception as e:
        print(e)
        return {"status": "error", "content": str(e)}
    
async def delete_knowledge_session(thread_id: str, s_knowledge_id: str):   
    try:
        vector_store = await get_vector_store_chroma(f"thread_{thread_id.replace('-', '_')}")
        
        collection = vector_store._collection
        collection_name = collection.name
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: collection.delete(where={"knowledge_id": s_knowledge_id})
        )
        
        knowledge_ids = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: collection.get(where={"thread_id": thread_id}, include=[])
        )
        
        if len(knowledge_ids["ids"]) == 0:
            client = vector_store._client
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.delete_collection(collection_name)
            )
        
        return {"status": "success", "content": f"Delete s_knowledge_id {s_knowledge_id} success."}
    
    except Exception as e:
        print(e)
        return {"status": "error", "content": str(e)}
   
async def delete_memory(user_id: str, memory_id: str):
    try:
        vector_store = await get_vector_store_chroma(f"user_{user_id.replace('-', '_')}")
        
        collection = vector_store._collection
        collection_name = collection.name
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: collection.delete(where={"memory_id": memory_id})
        )
        
        knowledge_ids = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: collection.get(where={"user_id": user_id}, include=[])
        )
        
        if len(knowledge_ids["ids"]) == 0:
            client = vector_store._client
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.delete_collection(collection_name)
            )
        
        return {"status": "success", "content": f"Delete memory_id {memory_id} success."}
    
    except Exception as e:
        print(e)
        return {"status": "error", "content": str(e)}