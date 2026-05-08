from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from google import genai

from typing import TypedDict, Annotated, Optional, Literal
from langchain.messages import AnyMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage, messages_to_dict, message_to_dict
from langchain_core.messages.utils import trim_messages
from langchain_core.runnables import RunnableConfig
from postgres.checkpoint.aio import AsyncPostgresSaver
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.types import Command
#from db_pool import get_db_pool

import operator, asyncio, os, fitz, uuid, base64, copy
from pydantic import BaseModel
from main_app import supabase_client
from dotenv import load_dotenv
from cryptography.hazmat.primitives.ciphers.aead import AESCCM
from fastapi import UploadFile

client = genai.Client()

DB_URI = os.getenv("DATABASE_URL")
ENC_KEY = base64.b64decode(os.getenv("KEY"))
google_api_key = os.getenv("GOOGLE_API_KEY")

pool = None #get_db_pool()

# We recommend using the following set of sampling parameters for generation

# Thinking mode for general tasks: temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
# Thinking mode for precise coding tasks (e.g. WebDev): temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0
# Instruct (or non-thinking) mode for general tasks: temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
# Instruct (or non-thinking) mode for reasoning tasks: temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
# Please note that the support for sampling parameters varies according to inference frameworks.

# Ada 3 pilihan: Auto, thinking, fast

# Google Gemini Free API
gemini = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=google_api_key,
    thinking_level="low",
    streaming=True,
)

gemini_instruct = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=google_api_key,
    thinking_level="medium",
    streaming=True,
)

gemini_thinking_reasoning = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=google_api_key,
    thinking_level="high",
    streaming=True,
)

gemini_embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=google_api_key,
)

# LLM-Instruct (VL included) (or basic) 
llm_instruct = ChatOpenAI(
    base_url="http://localhost:8000/v1", # https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/main
    api_key="none",
    model="/models/Qwen3.5-4B-Q4_K_M" # Instruct (or non-thinking) mode for general tasks
)

# LLM-Thinking (VL included) (LLM in reasoning task)
llm_thinking = ChatOpenAI(
    base_url="http://localhost:8001/v1", # https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/main
    api_key="none",
    model="/models/Qwen3.5-9B-Q4_K_M" # Thinking mode for general tasks
)

# Embedding-VL
embedding_vl = OpenAIEmbeddings(
    base_url="http://localhost:8002/v1", # https://huggingface.co/DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF/tree/main
    api_key="none",
    model="/models/Qwen3-4B-Embedding-Q4_K_M"
)

# reasoning decision
llm_reasoning = ChatOpenAI(
    base_url="http://localhost:8003/v1", # https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/main
    api_key="none",
    model="/models/Qwen3.5-4B-Q4_K_M" # Instruct (or non-thinking) mode for reasoning tasks
)

# Coding (or use reasoning)
llm_coding = ChatOpenAI(
    base_url="http://localhost:8004/v1", # https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/main
    api_key="none",
    model="/models/Qwen3.5-4B-Q4_K_M" # Thinking mode for precise coding tasks
)

# Image/video generation?


# State
def items_reducer(current: list, new: dict):
    if current is None:
        current = []
        
    result = copy.deepcopy(current)
    
    # Remove element
    for item in new.get("remove", []):
        if item in result:
            result.remove(item)
    
    # Append element
    for item in new.get("append", []):
        if item not in result:
            result.append(item)
            
    # Replace element
    for item in new.get("replace", []):
        _id = list(item.keys())[0]
        for i, existing in enumerate(result):
            if _id in existing:
                result[i] = item
                break
        
    return result    

class State(TypedDict):
    tenant_id: str
    user_id: str
    session_id: str # thread_id
    mode: str = "auto" # auto, thinking (reasoning), fast (no reasoning)

    messages: Annotated[list[AnyMessage], operator.add] = [] # list of AnyMessage, Human, AI, Tool, System
    selected_knowledge: Annotated[list[dict], items_reducer] = [] # list of dict: [{knowledge_id: {"metadata": metadata, "chunk_ids": [id_1, id_2]}}]
    chunk_knowledge: Annotated[list[dict], items_reducer] = [] # list of dict: [{chunk_id: content, "knowledge_id": knowledge_id}]
    retrieved_session_knowledge: Annotated[list[dict], items_reducer] = [] # list of dict: [{s_knowledge_id: {"metadata": metadata, "chunk_ids": [id_1, id_2]}}]
    chunk_retrieved_session_knowledge: Annotated[list[dict], items_reducer] = [] # list of dict: [{chunk_id: content, "s_knowledge_id": s_knowledge_id}]
    memory_ids: Annotated[list, items_reducer] = [] # list of str: [memory_id]
    memory: Annotated[list[dict], items_reducer] = [] # list of dict: [{memory_id: content}]
    
    iteration: int = 0 # For reasoning iteration
    route: str = "auto" # basic, coding-basic, coding-reasoning, thinking-reasoning. 
                        # Jika "auto", router akan memilih antara [basic, coding_basic, coding_reasoning, thinking_reasoning]

class router_output(BaseModel):
    mode: Literal["thinking", "fast"]
    route: Literal["basic", "coding_basic", "coding_reasoning", "thinking_reasoning"]


# Tools
## Tool: Put new memory
@tool
async def put_new_memory():
    """Put new memory to the database"""
    pass

## Tool: Put new knowledge_session (+ ttl)
@tool
async def put_new_knowledge_session():
    """Put new knowledge session to the database"""
    pass

## Tool: Fetch new knowledge (yang gak ada di state["selected_knowledge"])
@tool
async def fetch_new_knowledge():
    """Fetch new knowledge from the database"""
    pass

## Tool: Fetch new memory (yang gak ada di state["memory_ids"])
@tool
async def fetch_new_memory():
    """Fetch new memory from the database"""
    pass

## Tool: Fetch new knowledge_session (yang gak ada di state["retrieved_session_knowledge"])
@tool
async def fetch_new_knowledge_session():
    """Fetch new knowledge session from the database"""
    pass
    
## Tool: calculator, etc
@tool
async def calculator():
    """Use calculator to count number"""
    pass

## Tool: web search
@tool
async def web_search():
    """Use web search to find some external information"""
    pass

## Define Tools node
tools = [put_new_memory, put_new_knowledge_session, fetch_new_knowledge, fetch_new_memory, fetch_new_knowledge_session, calculator, web_search]

# llm_instruct = llm_instruct.bind_tools(tools)
# llm_thinking = llm_thinking.bind_tools(tools)
# llm_coding = llm_coding.bind_tools(tools)

gemini_instruct = gemini_instruct.bind_tools(tools)
gemini_thinking = gemini_thinking.bind_tools(tools)

tools_by_name = {tool.name: tool for tool in tools}

async def call_tools(state: State):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=tool_result,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}
    
async def should_continue(state: State):
    messages = state["messages"]
    mode = state["mode"]
    if not messages[-1].tool_calls:
        if mode == "thinking":
            return "reasoning"
        return state["route"] # basic or coding_basic
        
    return "call_tools"


# Agents
## Fetch history messages (dari checkpointer)
# async def fetch_history_messages(state: State):
    
    
    # # CUT MESSAGES
    
    # pass

## Fetch knowledge_session node if any
query_check_knowledge_session_ttl = """
SELECT s_knowledge_id
FROM Session_knowledges
WHERE s_knowledge_id = ANY($1);
"""

async def check_knowledge_session_ttl(state: State):
    """If there is no chunk from database, drop knowledge indices"""
    retrieved_session_knowledge = copy.deepcopy(state["retrieved_session_knowledge"])
    
    if len(retrieved_session_knowledge) == 0:
        return Command(goto="check_knowledge_exist")
    
    knowledge_ids = [list(s.keys())[0] for s in retrieved_session_knowledge]
    
    results = await pool.fetch(query_check_knowledge_session_ttl, knowledge_ids) # check if the document still exist in the database
    if len(results) > 0:
        fetched_knowledge_ids = [r["s_knowledge_id"] for r in results]
    else:
        fetched_knowledge_ids = []
    
    item_remove = []
    idx_remove = []
    for i, k_id in enumerate(knowledge_ids):
        if k_id not in fetched_knowledge_ids:
            idx_remove.append(i)
            item_remove.append(retrieved_session_knowledge[i])
            
    retrieved_session_knowledge = [x for i, x in enumerate(retrieved_session_knowledge) if i not in idx_remove]
            
    if len(retrieved_session_knowledge) == 0 and len(item_remove) > 0:
        return Command(
            goto="check_knowledge_exist",
            update={
                "retrieved_session_knowledge": {
                    "remove": item_remove,
                }
            }
        )
        
    else:
        return Command(
            goto = "fetch_knowledge_session",
            update={
                "retrieved_session_knowledge": {
                    "remove": item_remove,
                }
            }
        )

query_fetch_knowledge_session = """
SELECT s_knowledge_id, chunk_id, content
FROM Session_vectors
WHERE chunk_id = ANY($1);
"""

async def fetch_knowledge_session(state: State, config: RunnableConfig):
    """
    Fetch knowledge session from existing retrieved_session_knowledge indexes.
    """    
    # Load chunk_retrieved_session_knowledge
    chunk_ids = [c_id for k in state["retrieved_session_knowledge"] for k_id in k for c_id in k[k_id]["chunk_ids"]]
    
    chunk_knowledges = await pool.fetch(query_fetch_knowledge_session, chunk_ids)
    item_append = []
    for item in chunk_knowledges:
        s_knowledge_id = item["s_knowledge_id"]
        chunk_id = item["chunk_id"]
        content = decrypt(item["content"])
        item_append.append({chunk_id: content, "s_knowledge_id": s_knowledge_id})
    
    return {
        "chunk_retrieved_session_knowledge": {
            "append": item_append,
        }
    }
    
async def judge_knowledge_session(state: State):
    """
    Judge retrieved documents if there are still relevant or not based on history + prompt query
    """
    pass

## Fetch knowledge node if any
query_check_knowledge_exist = """
SELECT knowledge_id
FROM Knowledges
WHERE knowledge_id = ANY($1);
"""

async def check_knowledge_exist(state: State):
    """If there is no chunk from database, drop knowledge indices"""
    selected_knowledge = copy.deepcopy(state["selected_knowledge"])
    
    if len(selected_knowledge) == 0:
        return Command(goto="check_memory_exist")
    
    knowledge_ids = [list(s.keys())[0] for s in selected_knowledge]
    
    results = await pool.fetch(query_check_knowledge_exist, knowledge_ids) # check if the document still exist in the database
    if len(results) > 0:
        fetched_knowledge_ids = [r["knowledge_id"] for r in results]
    else:
        fetched_knowledge_ids = []
    
    item_remove = []
    idx_remove = []
    for i, k_id in enumerate(knowledge_ids):
        if k_id not in fetched_knowledge_ids:
            idx_remove.append(i)
            item_remove.append(selected_knowledge[i])
    
    selected_knowledge = [x for i, x in enumerate(selected_knowledge) if i not in idx_remove]
    
    if len(selected_knowledge) == 0 and len(item_remove) > 0:
        return Command(
            goto="check_memory_exist",
            update={
                "selected_knowledge": {
                    "remove": item_remove,
                }
            }
        )
        
    else:
        return Command(
            goto="fetch_knowledge",
            update={
                "selected_knowledge": {
                    "remove": item_remove,
                }
            }
        )

query_fetch_knowledge = """
SELECT knowledge_id, chunk_id, content
FROM Knowledge_vectors
WHERE chunk_id = ANY($1);
"""

async def fetch_knowledge(state: State):
    """
    Fetch knowledge from existing selected_knowledge indexes.
    """
    # Load chunk_knowledge
    chunk_ids = [c_id for k in state["selected_knowledge"] for k_id in k for c_id in k[k_id]["chunk_ids"]]
    
    chunk_knowledges = await pool.fetch(query_fetch_knowledge, chunk_ids)
    item_append = []
    for item in chunk_knowledges:
        knowledge_id = item["knowledge_id"]
        chunk_id = item["chunk_id"]
        content = decrypt(item["content"])
        item_append.append({chunk_id: content, "knowledge_id": knowledge_id})
    
    return {
        "chunk_knowledge": {
            "append": item_append,
        }
    }
    
async def judge_knowledge(state: State):
    """
    Judge retrieved documents if there are still relevant or not based on history + prompt query
    """
    pass

## Fetch memory node if any
query_memory_exist = """
SELECT memory_id, content
FROM User_memory
WHERE memory_id = ANY($1);
"""

async def check_memory_exist_and_fetch(state: State):
    """If there is no chunk from database, drop memory indices"""
    memory_ids = copy.deepcopy(state["memory_ids"])
    
    if len(memory_ids) == 0:
        return Command(goto="rag")
    
    results = await pool.fetch(query_memory_exist, memory_ids) # check if the memory still exist in the database
    if len(results) > 0:
        fetched_memory_ids = [r["memory_id"] for r in results]
    else:
        fetched_memory_ids = []
    
    item_remove = []
    idx_remove = []
    for i, m_id in enumerate(memory_ids):
        if m_id not in fetched_memory_ids:
            idx_remove.append(i)
            item_remove.append(memory_ids[i])
            
    memory_ids = [x for i, x in enumerate(memory_ids) if i not in idx_remove]
            
    if len(memory_ids) == 0 and len(item_remove) > 0:
        return Command(
            goto="rag",
            update={
                "memory_ids": {
                    "remove": item_remove,
                }
            }
        )
    
    else:
        item_append = []
        for item in results:
            memory_id = item["memory_id"]
            if memory_id in memory_ids:
                content = decrypt(item["content"])
                item_append.append({memory_id: content})
            
        return Command(
            goto="judge_memory",
            update={
                "memory_ids": {
                    "remove": item_remove,
                },
                "memory": {
                    "append": item_append,
                }
            }
        )

async def judge_memory(state: State):
    """
    Judge retrieved documents if there are still relevant or not based on history + prompt query
    """
    pass

## RAG (retrieve data from database based on just new query)
async def rag(state: State, config: RunnableConfig):
    tenant_id = config["configurable"]["tenant_id"]
    selected_knowledge = copy.deepcopy(state["selected_knowledge"]) # [{knowledge_id: {"metadata": metadata, "chunk_ids": [id_1, id_2]}}]
    knowledge_ids = [k_id for x in selected_knowledge for k_id in x]
    chunk_ids = [c_id for k in selected_knowledge for k_id in k for c_id in k[k_id]["chunk_ids"]]
    
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=gemini_instruct,
        max_tokens=8000,
        start_on="human",
        end_on=("human","tool"),
        include_system=True
    )
    
    if not isinstance(tenant_id, uuid.UUID):
        tenant_id = uuid.UUID(tenant_id)
    
    # Retrieve from database
    FETCH_CHUNKS = """
    SELECT chunk_id, knowledge_id, content
    FROM Knowledge_vectors
    WHERE tenant_id = $1
    ORDER BY embedding <=> $2
    LIMIT 5;
    """
    
    FETCH_KNOWLEDGE = """
    SELECT metadata
    FROM Knowledges
    WHERE tenant_id = $1 AND knowledge_id = ANY($2)
    ORDER BY array_position($2, knowledge_id);
    """
    
    system_query = f"""
    Based on history messages and the user's latest message,
    reformulate the user's question into a standalone question without need the history's contexts.
    
    This is the history messages:
    {messages_to_dict(messages[:-1])}
    
    Latest message: {message_to_dict(messages[-1])}
    
    Standalone question:"""
    
    llm_output = await gemini_instruct.ainvoke(SystemMessage(content=system_query))
    vector = await gemini_embedding.aembed_query(llm_output.content)
    
    async with pool.acquire() as conn:
        rows_fetch_chunks = await conn.fetch(FETCH_CHUNKS, tenant_id, vector)
        
        if len(rows_fetch_chunks) == 0:
            return Command(goto="router")
                
        knowledge_ids_from_chunks = list(set(x["knowledge_id"] for x in rows_fetch_chunks))
        
        rows_fetch_knowledges = await conn.fetch(FETCH_KNOWLEDGE, tenant_id, knowledge_ids_from_chunks)
        
        knowledge_id_metadata = {k_id: x["metadata"] for k_id, x in zip(knowledge_ids_from_chunks, rows_fetch_knowledges)}
    
    # Check if the knowledge is already retrieved on chunk_knowledge
    item_append_knowledge = {k_id: val for x in selected_knowledge for k_id, val in x.items()}
    for k_id, meta in knowledge_id_metadata.items():
        if k_id not in knowledge_ids:
            metadata = decrypt(meta)
            item_append_knowledge[k_id] = {"metadata": metadata, "chunk_ids": []}
    
    item_append = []
    ids_replace = set()
    for item in rows_fetch_chunks:
        chunk_id = item["chunk_id"]
        if chunk_id not in chunk_ids:
            content = decrypt(item["content"])
            knowledge_id = item["knowledge_id"] 
            item_append.append({chunk_id: content, "knowledge_id": knowledge_id})
            item_append_knowledge[knowledge_id]["chunk_ids"].append(chunk_id)
            
            if knowledge_id in knowledge_ids:
                ids_replace.add(knowledge_id)
            
    return Command(
        goto="router",
        update={
            "selected_knowledge": {
                "append": [{k_id: val} for k_id, val in item_append_knowledge.items() if k_id not in knowledge_ids],
                "replace": [{k_id: val} for k_id, val in item_append_knowledge.items() if k_id in ids_replace]
            },
            "chunk_knowledge": {
                "append": item_append,
            }
        }
    )

async def judge_rag(state: State):
    """
    Judge retrieved documents if there are still relevant or not based on history + prompt query
    """
    pass

router_model = gemini_instruct.with_structured_output(
    schema=router_output.model_json_schema(), method="json_schema"
)

## Agent: LLM-Instruct/router (fetch knowledge/knowledge_session/memory baru bila perlu)
async def router(state: State): # Tambahkan fungsi atau state untuk format output yang tetap {route: "", mode: ""}
    """Jika prompt user terkait dengan (dokumen) perusahaan, fetch_knowledge. Jika diminta mengingat, fetch user_memory."""
    # Router menentukan mode "thinking" atau "fast" sesuai input user.
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=gemini_instruct,
        max_tokens=8000,
        start_on="human",
        end_on=("human","tool"),
        include_system=True
    )
    
    system_query = f"""
    Based on the messages history, consider which mode and route the 
    
    This is the history messages:
    {messages_to_dict(messages)}
    
    """
    
    response = router_model.ainvoke(SystemMessage(content=system_query))
    
    return {"mode": response["mode"], "route": response["route"]}
    
## Agent: Basic (same as router model), Visual (Photo, Video) Analysis (non-thinking)
async def basic(state: State):
    pass

## Agent: Coding basic
async def coding_basic(state: State):
    pass
    
## Agent: Coding reasoning
async def coding_reasoning(state: State):
    pass
    
## Agent: Coding end (conclusion)
async def coding_end(state: State):
    pass
    
## Agent: Thinking reasoning
async def thinking_reasoning(state: State):
    pass
    
## Agent: Thinking end (conclusion)
async def thinking_end(state: State):
    pass
    
## Reasoning node untuk Agent Thinking dan Coding (untuk pengembangan: tambahkan interrupt dan/atau user input sebelum masuk reasoning node)
async def reasoning(state: State):
    route = state["route"] # coding or thinking
    return route
 
## Agent: Report Maker (Orchestration)

## Agent: 


# Define agent
async def agent(db_pool, input_data: dict, f: Optional[UploadFile] = None):
    global pool
    pool = db_pool
    builder = StateGraph(State)
    
    builder.add_node("check_knowledge_session_ttl", check_knowledge_session_ttl)
    builder.add_node("fetch_knowledge_session", fetch_knowledge_session)
    builder.add_node("judge_knowledge_session", judge_knowledge_session)
    builder.add_node("check_knowledge_exist", check_knowledge_exist)
    builder.add_node("fetch_knowledge", fetch_knowledge)
    builder.add_node("judge_knowledge", judge_knowledge)
    builder.add_node("check_memory_exist_and_fetch", check_memory_exist_and_fetch)
    # builder.add_node("fetch_memory", fetch_memory)
    builder.add_node("judge_memory", judge_memory)
    #builder.add_node("fetch_history_messages", fetch_history_messages)
    # builder.add_node("chunk_knowledge_session", chunk_knowledge_session)
    builder.add_node("rag", rag)
    builder.add_node("router", router)
    builder.add_node("basic", basic)
    builder.add_node("coding_basic", coding_basic)
    builder.add_node("coding_reasoning", coding_reasoning)
    builder.add_node("coding_end", coding_end)
    builder.add_node("thinking_reasoning", thinking_reasoning)
    builder.add_node("thinking_end", thinking_end)
    builder.add_node("reasoning", reasoning)
    builder.add_node("call_tools", call_tools)
    
    
    builder.add_edge(START, "check_knowledge_session_ttl")
    builder.add_edge("check_knowledge_session_ttl", "fetch_knowledge_session")
    builder.add_edge("check_knowledge_session_ttl", "check_knowledge_exist")
    builder.add_edge("fetch_knowledge_session", "judge_knowledge_session")
    builder.add_edge("judge_knowledge_session", "check_knowledge_exist")

    builder.add_edge("check_knowledge_exist", "fetch_knowledge")
    builder.add_edge("check_knowledge_exist", "check_memory_exist")
    builder.add_edge("fetch_knowledge", "judge_knowledge")
    builder.add_edge("judge_knowledge", "check_memory_exist")
    
    builder.add_edge("check_memory_exist_and_fetch", "rag")
    builder.add_edge("check_memory_exist_and_fetch", "judge_memory")
    builder.add_edge("judge_memory", "rag")
    # builder.add_conditional_edges("check_memory_exist", check_any_documents_uploaded, ["chunk_knowledge_session", "router"])
    # builder.add_conditional_edges("fetch_memory", check_any_documents_uploaded, ["chunk_knowledge_session", "router"])
    # builder.add_edge("check_memory_exist", "fetch_history_messages")
    # builder.add_edge("fetch_memory", "fetch_history_messages")

    #builder.add_conditional_edges("fetch_history_messages", check_any_documents_uploaded, ["chunk_knowledge_session", "router"])
    # builder.add_edge("chunk_knowledge_session", "router")
    builder.add_edge("rag", "router")
    builder.add_conditional_edges("router", lambda s: s["route"], ["basic", "coding_basic", "coding_reasoning", "thinking_reasoning"])
    
    # non-reasoning (basic and coding)
    builder.add_conditional_edges("basic", should_continue, ["call_tools", END])
    builder.add_edge("call_tools", "basic")
    
    builder.add_conditional_edges("coding_basic", should_continue, ["call_tools", END])
    builder.add_edge("call_tools", "coding_basic")
    
    # thinking-reasoning (coding and thinking)
    builder.add_conditional_edges("reasoning", lambda s: s["route"], ["coding_reasoning", "coding_end", "thinking_reasoning", "thinking_end"])
    ## coding
    builder.add_conditional_edges("coding_reasoning", should_continue, ["call_tools", "reasoning"])
    builder.add_edge("call_tools", "coding_reasoning")
    ## thinking
    builder.add_conditional_edges("thinking_reasoning", should_continue, ["call_tools", "reasoning"])
    builder.add_edge("call_tools", "thinking_reasoning")
    
    thread_id = input_data["thread_id"]
    user_id = input_data["user_id"]
    tenant_id = input_data["tenant_id"]
    input_prompt = input_data["input_prompt"]
    
    config = {
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
                {"messages": [{"role": "user", "content": input_prompt}]},
                config,
                stream_mode="values",
                version="v2",
            ):
                if chunk["type"] == "messages":
                    msg, metadata = chunk["data"]
                    if msg.content:
                        yield msg.content #chunk["messages"][-1]

    # png_graph = agent.get_graph().draw_mermaid_png()
    # with open("graph2.png", "wb") as f:
        # f.write(png_graph)

aesccm = AESCCM(ENC_KEY)

def encrypt(text: str) -> bytes:
    if not isinstance(text, str):
        text = str(text)
        
    nonce = os.urandom(13)
    ciphertext = aesccm.encrypt(nonce, text.encode(), None)
    
    return nonce + ciphertext
    
def decrypt(data: bytes) -> str:
    nonce = data[:13]
    ciphertext = data[13:]
    plaintext = aesccm.decrypt(nonce, ciphertext, None)
    
    return plaintext.decode()
 
 async def chunk_document(f, file_bytes):
    filetype_map = {
        "application/pdf": "pdf",
        "application/epub+zip": "epub",
        "text/plain": "txt",
    }
    
    filetype = filetype_map.get(f.content_type, "pdf")
    doc = fitz.open(stream=file_bytes, filetype=filetype)
    
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
        
    return splitter.split_text(text), len(doc)
 
## Upload user_document per chat, add TTL
async def save_chunks_session_to_db(pool, chunks, pages, f, configurable, prompt):
    def to_uuid(value: str | uuid.UUID) -> uuid.UUID:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(value)
    
    user_id = to_uuid(configurable["user_id"])
    tenant_id = to_uuid(configurable["tenant_id"])
    thread_id = to_uuid(configurable["thread_id"])
    
    s_knowledge_id = uuid.uuid4()
    filename = f.filename
    content_type = f.content_type
    
    metadata = {
        "filename": filename,
        "content-type": content_type,
        "pages": pages,
    }
    
    vector = gemini_embedding.embed_documents(chunks)
    # vector_prompt = gemini_embedding.embed_query(prompt)

    records = [
        (uuid.uuid4(), s_knowledge_id, tenant_id, user_id, encrypt(chunk), vec)
        for chunk, vec in zip(chunks, vector)
    ]
    
    chunk_ids = [r[0] for r in records]
    
    # FETCH_CHUNKS = """
    # SELECT chunk_id, s_knowledge_id, content
    # FROM Session_vectors
    # WHERE tenant_id = $1 AND user_id = $2 AND thread_id = $3
    # ORDER BY embedding <=> $4
    # LIMIT 5;
    # """
    
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO Session_nowledges (s_knowledge_id, tenant_id, user_id, metadata)
                VALUES ($1, $2, $3, $4)
                """,
                s_knowledge_id, tenant_id, user_id, encrypt(metadata)
            )
            
            await conn.executemany(
                """
                INSERT INTO Session_vectors (chunk_id, s_knowledge_id, tenant_id, user_id, content, embedding)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                records
            )
            
            # rows_fetch_chunks = await conn.fetch(FETCH_CHUNKS, tenant_id, vector_prompt)
            
    except Exception as e:
        print(e)
        return {"s_knowledge_id": 0, "user_id": 0}
        
    return {"s_knowledge_id": s_knowledge_id, "user_id": user_id, "metadata": metadata, "chunk_ids": chunk_ids}

async def put_new_knowledge_session(pool, f, config, prompt):
    file_bytes = await f.read()
    
    configurable = config["configurable"]
    user_id = configurable["user_id"]
    
    # Upload file to storage
    try:
        response = (
            supabase_client.storage.from_("knowledge_session").upload(
                file=file_bytes,
                path=f"{str(user_id)}/{f.filename}",
                file_options={
                    "content-type": f.content_type,
                    "upsert": "false"
                }
            )
        )
    except Exception as e:
        print(e)
    
    # Chunking document
    chunks, pages = await chunk_document(f, file_bytes)
    
    result = await save_chunks_session_to_db(pool, chunks, pages, f, configurable, prompt)
    
    return result

## Tool: Put new knowledge (by tenant admin)
async def save_chunks_to_db(pool, chunks, pages, f, tenant_id):
    if not isinstance(tenant_id, uuid.UUID):
        tenant_id = uuid.UUID(tenant_id)
    
    knowledge_id = uuid.uuid4()
    filename = f.filename
    content_type = f.content_type
    
    metadata = {
        "filename": filename,
        "content-type": content_type,
        "pages": pages,
    }
    
    vector = gemini_embedding.embed_documents(chunks)

    records = [
        (uuid.uuid4(), knowledge_id, tenant_id, encrypt(chunk), vec)
        for chunk, vec in zip(chunks, vector)
    ]
    
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO Knowledges (knowledge_id, tenant_id, metadata)
                VALUES ($1, $2, $3)
                """,
                knowledge_id, tenant_id, encrypt(metadata)
            )
            
            await conn.executemany(
                """
                INSERT INTO Knowledge_vectors (chunk_id, knowledge_id, tenant_id, content, embedding)
                VALUES ($1, $2, $3, $4, $5)
                """,
                records
            )

    except Exception as e:
        print(e)
        return {"knowledge_id": 0, "tenant_id": 0}
        
    return {"knowledge_id": knowledge_id, "tenant_id": tenant_id}

async def put_new_knowledge(db_pool, f, tenant_id):
    global pool
    pool = db_pool
    
    file_bytes = await f.read()
    
    # Upload file to storage
    try:
        response = (
            supabase_client.storage.from_("knowledges").upload(
                file=file_bytes,
                path=f"{str(tenant_id)}/{f.filename}",
                file_options={
                    "content-type": f.content_type,
                    "upsert": "false"
                }
            )
        )
    except Exception as e:
        print(e)
    
    # Chunking document
    chunks, pages = await chunk_document(f, file_bytes)
    
    result = await save_chunks_to_db(pool, chunks, pages, f, tenant_id)
  
    return result

# if __name__ == "__main__":
    # asyncio.run(agent())


# For next development
# Use cron job to update state of the retrieved document ids. delete the id that will have been remove in the State
# Paralellize retrive knowledge/memory