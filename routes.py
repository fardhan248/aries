from fastapi import Request, APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from langgraph_core import streaming, put_new_knowledge, chat, get_agent_graph
from typing import Optional
import uuid

router = APIRouter()

@router.get("/")
async def root():
    return {"message": "Chatbot Nusantara is running"}

@router.get("/hello")
async def hello():
    return {"Hello, World!"}

# @router.get("/get_graph")
# async def get_graph(request: Request, input_data: dict):
    # pool = request.app.state.pool
    # for chunk in agent(pool, input_data):
        # return {"messages": chunk}

# Di sini kumpulan router api langgraph

# Stream langgraph
@router.post("/stream_chat")
async def stream_chat(request: Request, input_data: dict, f: Optional[UploadFile] = File(None)):
    pool = request.app.state.pool
    
    if f:
        return StreamingResponse(
            streaming(pool, input_data, f), 
            media_type="text/plain",
        )
    else:
        return StreamingResponse(
            streaming(pool, input_data), 
            media_type="text/plain",
        )

@router.post("/chat")
async def chat(request: Request, input_data: dict, f: Optional[UploadFile] = File(None)):
    pool = request.app.state.pool
    
    if f:
        return chat(pool, input_data, f)
    else:
        return chat(pool, input_data)
        
@router.get("/get_graph")
async def get_graph():
    await get_agent_graph()
    return {"success"}

# Get Chat Session (and the history)

# Get user_id from log in

# Delete Chat Session

# Upload document (RAG)
@router.post("/upload")
async def upload(request: Request, f: UploadFile, tenant_id: uuid.UUID):
    pool = request.app.state.pool
    
    return await put_new_knowledge(pool, f, tenant_id)

# Delete document (RAG)
