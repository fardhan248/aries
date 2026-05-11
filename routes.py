from fastapi import Request, APIRouter, UploadFile, File, Form, Body
from fastapi.responses import StreamingResponse
from src.chat_completion import streaming, chat_workflow, get_agent_graph
from utils.documents_utils import put_new_knowledge
from typing import Optional
from src.add_member import new_company, new_user
from body_models.router_models import AddUserInput, ChatInput
from utils.health_check import health_check
import uuid, json

router = APIRouter()

@router.get("/") #✅
async def root():
    return {"message": "Chatbot Nusantara is running"}


@router.get("/hello") #✅
async def hello():
    return {"message": "Hello, World!"}


# Stream langgraph
@router.post("/chat/stream_chat/{thread_id}")
async def stream_chat(
    request: Request, 
    thread_id: uuid.UUID, 
    input_data: str = Form(...), 
    f: Optional[UploadFile] = File(None),
):
    pool = request.app.state.pool
    
    input_data = ChatInput(**json.loads(input_data))
    
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

@router.post("/chat/{thread_id}")
async def chat(
    request: Request, 
    thread_id: uuid.UUID, 
    input_data: str = Form(...), 
    f: Optional[UploadFile] = File(None),
):
    pool = request.app.state.pool
    
    input_data = ChatInput(**json.loads(input_data))
    
    if f:
        return await chat_workflow(pool, input_data, f)
    else:
        return await chat_workflow(pool, input_data)
  
    
@router.post("/add_member/add_company") #✅
async def add_company(request: Request, tenant: str = Body(...)):
    pool = request.app.state.pool

    return await new_company(pool, tenant)
    
@router.post("/add_member/{tenant_id}/add_user/") #✅
async def add_user(
    request: Request, 
    tenant_id: uuid.UUID, 
    input_data: AddUserInput,
):
    pool = request.app.state.pool

    user = input_data.user
    role = input_data.role

    return await new_user(pool, tenant_id, user, role)


# Upload document (RAG)
@router.post("/upload/{tenant_id}") #✅
async def upload(request: Request, tenant_id: uuid.UUID, f: UploadFile):
    pool = request.app.state.pool
    
    return await put_new_knowledge(pool, f, tenant_id)


@router.get("/get_graph") #✅
async def get_graph():
    await get_agent_graph()
    return {"status": "success"}
    

@router.get("/health") #✅
async def health(request: Request):
    pool = request.app.state.pool
    
    return await health_check(pool)

# Get Chat Session (and the history)

# Get user_id from log in

# Delete Chat Session

# Delete document (RAG)