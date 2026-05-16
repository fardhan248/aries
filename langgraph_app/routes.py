from fastapi import Request, APIRouter, UploadFile, File, Form, Body
from fastapi.responses import StreamingResponse
from src.chat_completion import streaming, get_agent_graph, chat_workflow
from utils.documents_utils import put_new_knowledge
from typing import Optional
from src.add_member import new_company, new_user, new_chat
from body_models.router_models import AddUserInput, ChatInput#, NewChat
from utils.health_check import health_check
from starlette.datastructures import Headers
import uuid, json, io

router = APIRouter()

@router.get("/") #✅
async def root():
    return {"message": "Chatbot Nusantara is running"}


@router.get("/hello") #✅
async def hello():
    return {"message": "Hello, World!"}


# Chat Endpoints
@router.post("/chat/new") #✅
async def make_new_chat(
    request: Request,
    input_data: str = Form(...),
    f: Optional[UploadFile] = File(None),
):
    pool = request.app.state.pool
    
    input_data = ChatInput(**json.loads(input_data))
    
    if f is not None:
        file_content = await f.read()
        f2 = UploadFile(
            filename=f.filename,
            file=io.BytesIO(file_content),
            headers=Headers({"content_type": f.content_type}),
            size=f.size,
        )
    else:   
        f2=f
    
    if input_data.thread_id is not None:    
        return await chat(
                request=request,
                thread_id=input_data.thread_id,
                input_data=input_data.model_dump_json(),
                f=f2,
            )
    
    result = await new_chat(pool, input_data)
    
    if result["status"] == "error":
        return result
       
    input_data.thread_id = result["thread_id"]
        
    return await chat(
            request=request,
            thread_id=input_data.thread_id,
            input_data=input_data.model_dump_json(),
            f=f2,
        )

@router.post("/chat/{thread_id}") #✅
async def chat(
    request: Request, 
    thread_id: uuid.UUID, 
    input_data: str = Form(...), 
    f: Optional[UploadFile] = File(None),
):
    pool = request.app.state.pool

    input_data = ChatInput(**json.loads(input_data))
    streaming = input_data.streaming
    
    if input_data.thread_id is None and thread_id is not None:
        input_data.thread_id = thread_id
    
    if streaming == False:
        return await chat_workflow(pool, input_data, f)
    
    else:
        return StreamingResponse(
            streaming(pool, input_data, f), 
            media_type="text/event_stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    
@router.post("/add_member/add_company") #✅
async def add_company(request: Request, tenant: str = Body(...)):
    pool = request.app.state.pool

    return await new_company(pool, tenant)
    
@router.post("/add_member/{tenant_id}/add_user") #✅
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
    
    return await put_new_knowledge(f, tenant_id)


# @router.get("/get_graph") #✅
# async def get_graph():
    # await get_agent_graph()
    # return {"status": "success"}
    

@router.get("/health") #✅
async def health(request: Request):
    pool = request.app.state.pool
    
    return await health_check(pool)

# Get Chat Session (and the history)

# Get user_id from log in

# Delete Chat Session

# Delete document (RAG)