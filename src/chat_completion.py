from postgres.checkpoint.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage, message_to_dict
from langchain_core.runnables import RunnableConfig

import os, uuid
from models.gemini import gemini
from utils.documents_utils import put_new_knowledge_session
from fastapi import UploadFile
from src.langgraph_core import get_agent
from dotenv import load_dotenv
from typing_extensions import Optional
from utils.documents_utils import encrypt, decrypt
from string_utils.database_queries import TitleQueries
from string_utils.prompts import PromptTitle

load_dotenv()

DB_URI = os.getenv("DATABASE_URL")
pool = None 

titlequeries = TitleQueries()
prompttitle = PromptTitle()

async def new_chat(tenant_id: uuid.UUID, user_id: uuid.UUID, thread_id: uuid.UUID, input_prompt: str):
    try:    
        async with pool.acquire() as conn:
            response = await conn.fetchrow(
                titlequeries.FETCH_SESSION_TITLE,
                thread_id,
            ) 
            
            title = response["title"]
            decrypted_title = await decrypt(title)
            
            if response is not None:
                return {"status": "success", "content": "Already in database", "title": decrypted_title}
            
            thread_id = uuid.uuid4()
            
            SESSION_TITLE_SYSTEM_QUERY = prompttitle.SESSION_TITLE_SYSTEM_QUERY.format_map({
                "input_prompt": input_prompt,
            })
            
            title = gemini.ainvoke(SESSION_TITLE_SYSTEM_QUERY)
            encrypted_title = await encrypt(title)
            
            await conn.execute(
                titlequeries.INSERT_SESSION_TITLE,
                thread_id, tenant_id, user_id, encrypted_title
            )                 
            
            return {"status": "success", "content": f"Session created {title}", "title": title}
    
    except Exception as e:
        print(e)
        return {"status": "error", "content": str(e), "title": None}

async def streaming(db_pool, input_data: dict, f: Optional[UploadFile] = None): 
    # use get_stream_writer: https://reference.langchain.com/python/langgraph/config/get_stream_writer
    global pool
    pool = db_pool
    
    builder = await get_agent()
    
    thread_id = input_data["thread_id"]
    user_id = input_data["user_id"]
    tenant_id = input_data["tenant_id"]
    input_prompt = input_data["input_prompt"]
    mode = input_data["mode"]
    
    check_session = await new_chat(tenant_id, user_id, thread_id, input_prompt)
    title = check_session["title"]
    
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
        }
    }
    
    if f is not None:
        result = await put_new_knowledge_session(pool, f, config, input_prompt)
        
        if result["s_knowledge_id"] != 0:
            print("Success store new document")
        else:
            print("Error while store new document")
    else:
        result = None
    
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        agent = builder.compile(checkpointer=checkpointer) # Don't forget use checkpointer
        
        if result is not None:
            if result.get("metadata", None) is not None:
                await agent.aupdate_state(
                    config,
                    {
                        "retrieved_session_knowledge": {
                            "append": [{result["s_knowledge_id"]: {"metadata": result["result"], "chunk_ids": result["chunk_ids"]}}]
                        }
                    }
                )
        
        async for chunk in agent.astream(
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "thread_id": thread_id,
                "messages": [HumanMessage(content=input_prompt)],
                "title": title,
                "mode": mode, 
                "streaming_mode": True,
            },
            config,
            stream_mode="custom",
            version="v2",
        ):
            if chunk["type"] == "messages":
                msg, metadata = chunk["data"]
                if msg.content:
                    yield msg.content 

async def chat_workflow(db_pool, input_data: dict, f: Optional[UploadFile] = None):
    global pool
    pool = db_pool
    
    builder = await get_agent()
    
    thread_id = input_data["thread_id"]
    user_id = input_data["user_id"]
    tenant_id = input_data["tenant_id"]
    input_prompt = input_data["input_prompt"]
    mode = input_data["mode"]
    
    check_session = await new_chat(tenant_id, user_id, thread_id, input_prompt)
    title = check_session["title"]
    
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
        }
    }
    
    if f is not None:
        result = await put_new_knowledge_session(pool, f, config, input_prompt)
        
        if result["s_knowledge_id"] != 0:
            print("Success store new document")
        else:
            print("Error while store new document")
    else:
        result = None
    
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        agent = builder.compile(checkpointer=checkpointer) # Don't forget use checkpointer
        
        if result is not None:
            if result.get("metadata", None) is not None:
                await agent.aupdate_state(
                    config,
                    {
                        "retrieved_session_knowledge": {
                            "append": [{result["s_knowledge_id"]: {"metadata": result["result"], "chunk_ids": result["chunk_ids"]}}]
                        }
                    }
                )
                
        result = agent.ainvoke(
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "thread_id": thread_id,
                "messages": [HumanMessage(content=input_prompt)],
                "title": title,
                "mode": mode, 
                "streaming_mode": False,
            },
            config,
        )
        
    return message_to_dict(result["message"][-1])

async def get_agent_graph():
    builder = await get_agent()
    
    agent = builder.compile()
    
    png_graph = agent.get_graph().draw_mermaid_png()
    with open("output/graph.png", "wb") as f:
        f.write(png_graph)