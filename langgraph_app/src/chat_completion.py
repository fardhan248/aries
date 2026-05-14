from postgres.checkpoint.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage, message_to_dict
from langchain_core.runnables import RunnableConfig

import os, uuid, logging, traceback
from models.gemini import gemini
from utils.documents_utils import put_new_knowledge_session
from fastapi import UploadFile
from src.langgraph_core import get_agent
from dotenv import load_dotenv
from typing_extensions import Optional
from body_models.router_models import ChatInput

import src.langgraph_core as lang_core

load_dotenv()

DB_URI = os.getenv("DATABASE_URL")
pool = None 

logger = logging.getLogger(__name__)

async def streaming(db_pool, input_data: ChatInput, f: Optional[UploadFile] = None): 
    # use get_stream_writer: https://reference.langchain.com/python/langgraph/config/get_stream_writer
    global pool
    pool = db_pool
    lang_core.pool = pool
    
    builder = await get_agent()
    
    thread_id = input_data.thread_id
    user_id = input_data.user_id
    tenant_id = input_data.tenant_id
    input_prompt = input_data.input_prompt
    mode = input_data.mode
    
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
        }
    }
    
    if f is not None:
        print("Upload file")
        result = await put_new_knowledge_session(pool, f, config, input_prompt)
        
        if result["s_knowledge_id"] != 0:
            print("Success store new document")
        else:
            print("Error while store new document")
    else:
        result = None
    
    try:
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            agent = builder.compile(checkpointer=checkpointer) # Don't forget use checkpointer
            
            if result is not None:
                if result.get("metadata", None) is not None:
                    async for chunk in agent.astream(
                        {
                            "tenant_id": str(tenant_id),
                            "user_id": str(user_id),
                            "thread_id": str(thread_id),
                            "messages": [HumanMessage(content=input_prompt)],
                            "mode": mode, 
                            "streaming_mode": True,
                            "retrieved_session_knowledge": {
                                "append": [{result["s_knowledge_id"]: {"metadata": result["metadata"], "chunk_ids": result["chunk_ids"]}}],
                            },
                        },
                        config,
                        stream_mode="custom",
                        version="v2",
                    ):
                        if chunk["type"] == "messages":
                            msg, metadata = chunk["data"]
                            if msg.content:
                                yield msg.content 
                                    
                else:
                    yield result
            
            else:
                async for chunk in agent.astream(
                    {
                        "tenant_id": str(tenant_id),
                        "user_id": str(user_id),
                        "thread_id": str(thread_id),
                        "messages": [HumanMessage(content=input_prompt)],
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
    except Exception as e:
        print(e)
        yield {"status": "error", "content": str(e)}

async def chat_workflow(db_pool, input_data: dict, f: Optional[UploadFile] = None):
    global pool
    pool = db_pool
    lang_core.pool = pool
    
    builder = await get_agent()
    
    thread_id = input_data.thread_id
    user_id = input_data.user_id
    tenant_id = input_data.tenant_id
    input_prompt = input_data.input_prompt
    mode = input_data.mode
    
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
        }
    }
    
    if f is not None:
        print("Upload file")
        result = await put_new_knowledge_session(pool, f, config, input_prompt)
        
        if result["s_knowledge_id"] != 0:
            print("Success store new document")
        else:
            print("Error while store new document")
            return {"status": "error", "content": result["content"]}
    else:
        result = None
    
    try:
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            agent = builder.compile(checkpointer=checkpointer) # Don't forget use checkpointer
            
            if result is not None:
                if result.get("metadata", None) is not None:
                    result_agent = await agent.ainvoke(
                        {
                            "tenant_id": str(tenant_id),
                            "user_id": str(user_id),
                            "thread_id": str(thread_id),
                            "messages": [HumanMessage(content=input_prompt)],
                            "mode": mode, 
                            "streaming_mode": False,
                            "retrieved_session_knowledge": {
                                "append": [{result["s_knowledge_id"]: {"metadata": result["metadata"], "chunk_ids": result["chunk_ids"]}}],
                            },
                        },
                        config,
                    )
                    
                    try:
                        text = message_to_dict(result_agent["messages"][-1])["data"]["content"][0]["text"]
                    except Exception as e:
                        print(e)
                        text = message_to_dict(result_agent["messages"][-1])
                    
                    return {"thread_id": str(thread_id), "content": text}
                        
                else:
                    return result
                    
            else:
                result_agent = await agent.ainvoke(
                    {
                        "tenant_id": str(tenant_id),
                        "user_id": str(user_id),
                        "thread_id": str(thread_id),
                        "messages": [HumanMessage(content=input_prompt)],
                        "mode": mode, 
                        "streaming_mode": False,
                    },
                    config,
                )
                
                try:
                    text = message_to_dict(result_agent["messages"][-1])["data"]["content"][0]["text"]
                except Exception as e:
                    print(e)
                    text = message_to_dict(result_agent["messages"][-1])

                return {"thread_id": str(thread_id), "content": text}
    except Exception as e:
        logger.error(traceback.format_exc())
        print(e)
        return {"status": "error", "content": str(e)}
                
async def get_agent_graph():
    builder = await get_agent()
    
    agent = builder.compile()
    
    png_graph = agent.get_graph().draw_mermaid_png()
    with open("output/graph.png", "wb") as f:
        f.write(png_graph)