from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from google import genai

from typing_extensions import TypedDict, Annotated, Optional, Literal
from langchain.messages import AnyMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage, messages_to_dict, message_to_dict, AIMessage
from langchain_core.messages.utils import trim_messages
from langchain_core.runnables import RunnableConfig
from postgres.checkpoint.aio import AsyncPostgresSaver
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.types import Command
from langgraph.prebuilt import InjectedState #, InjectedToolCallId
from langchain_core.tools import InjectedToolCallId
#from db_pool import get_db_pool

import operator, asyncio, os, fitz, uuid, base64, copy
from pydantic import BaseModel
# from main_app import supabase_client
from contextmanager_utils import supabase_client
from dotenv import load_dotenv
from cryptography.hazmat.primitives.ciphers.aead import AESCCM
from fastapi import UploadFile
from collections import defaultdict

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
        if isinstance(item, dict):
            _id = list(item.keys())[0]
            for i, existing in enumerate(result):
                if _id in existing:
                    result[i] = item
                    break
        else:
            for i, existing in enumerate(result):
                result[i] = item
        
    return result    

class State(TypedDict):
    tenant_id: str
    user_id: str
    session_id: str # thread_id
    mode: str = "auto" # auto, thinking (reasoning), fast (no reasoning)
    streaming_mode: str = True

    messages: Annotated[list[AnyMessage], operator.add] = [] # list of AnyMessage, Human, AI, Tool, System
    selected_knowledge: Annotated[list[dict], items_reducer] = [] # list of dict: [{knowledge_id: {"metadata": metadata, "chunk_ids": [id_1, id_2]}}]
    chunk_knowledge: Annotated[list[dict], items_reducer] = [] # list of dict: [{chunk_id: content, "knowledge_id": knowledge_id}]
    retrieved_session_knowledge: Annotated[list[dict], items_reducer] = [] # list of dict: [{s_knowledge_id: {"metadata": metadata, "chunk_ids": [id_1, id_2]}}]
    chunk_retrieved_session_knowledge: Annotated[list[dict], items_reducer] = [] # list of dict: [{chunk_id: content, "s_knowledge_id": s_knowledge_id}]
    memory_ids: Annotated[list, items_reducer] = [] # list of str: [memory_id]
    memory: Annotated[list[dict], items_reducer] = [] # list of dict: [{memory_id: content}]
    
    last_query: str
    iteration: int = 0 # For reasoning iteration
    route: str # basic, coding-basic, coding-reasoning, thinking-reasoning. 
               # router akan memilih antara [basic, coding_basic, coding_react, thinking_react]
    reasoning_questions_obervation: Annotated[list[dict[str, str]], operator.add] = []
    
    

class router_output(BaseModel):
    route: Literal["basic", "coding_basic", "coding_react", "thinking_react"]


# Tools
## Tool: Put new memory
@tool
async def put_new_memory(
    query: str, 
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str | Command: # return: success put new user memory to the database
    """
    Store a new memory entry for the user into the database.

    Checks if the user's stored memories have reached the limit (20 entries).
    If not, encrypts the content, generates an embedding vector, and inserts
    the new memory record into the User_memory table.

    Args:
        query (str): The memory content to store.

    Returns:
        Command: Updates memory_ids and memory state on success.
        str: Success or failure message.
    """
    vector = await gemini_embedding.aembed_documents([query])
    user_id = state["user_id"]
    tenant_id = state["tenant_id"]
    memory_id = uuid.uuid4()
    
    CHECK_MEMORY = """
    SELECT memory_id
    FROM User_memory
    WHERE user_id = $1;
    """
    
    PUT_NEW_MEMORY = """
    INSERT INTO User_memory (memory_id, tenant_id, user_id, content, embedding)
    VALUES ($1, $2, $3, $4, $5)
    """
    
    try:
        async with pool.acquire() as conn:
            memories = await conn.fetch(CHECK_MEMORY, user_id)
            
            if len(memories) >= 20:
                return "Failed put new memory to the database. The user's saved memory has reached limit memory: 20" 
                
            await conn.execute(PUT_NEW_MEMORY, memory_id, tenant_id, user_id, encrypt(query), vector[0])
            
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="success put new user memory to the database.", 
                            tool_call_id=tool_call_id,
                            name="put_new_memory",
                        )
                    ],
                    "memory_ids": {
                        "append": [memory_id]
                    },
                    "memory": {
                        "append": [{memory_id: query}]
                    },
                }
            )
            
    except Exception as e:
        print(e)
        return "Failed put new memory to the database." 
        
## Tool: Put new knowledge_session (+ ttl) (udah ada fungsi eksternal)
# @tool
# async def put_new_knowledge_session():
    # """Put new knowledge session to the database"""
    # pass

## Tool: Fetch new knowledge (yang gak ada di state["selected_knowledge"]) 
@tool
async def fetch_new_knowledge(
    query: str, 
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command | str:
    """
    Fetch new knowledge chunks from the database that are not yet in the current state.

    Embeds the query, retrieves the top 5 closest knowledge chunks via vector similarity search,
    filters out chunks already present in the state, fetches metadata for newly found knowledge entries,
    and updates the state with the new knowledge and chunks.

    Args:
        query (str): The search query to embed and use for similarity search.

    Returns:
        Command: Updates selected_knowledge and chunk_knowledge state on success.
        str: Error message on failure.
    """
    tenant_id = state["tenant_id"]
    chunk_ids = [c_id for k in state["selected_knowledge"] for k_id in k for c_id in k[k_id]["chunk_ids"]]    
    selected_knowledge_dict = {k_id: val for item in state["selected_knowledge"] for k_id, val in item.items()}
    
    FETCH_NEW_KNOWLEDGE_CHUNK = """
    SELECT chunk_id, knowledge_id, content
    FROM Knowledge_vectors
    WHERE tenant_id = $1
    ORDER BY embedding <=> $2
    LIMIT 5;
    """
    
    FETCH_NEW_KNOWLEDGE = """
    SELECT knowledge_id, metadata
    FROM Knowledges
    WHERE tenant_id = $1 AND knowledge_id = ANY($2);
    """
    
    vector = await gemini_embedding.aembed_query(query)
    
    try:
        async with pool.acquire() as conn:
            result_chunk = await conn.fetch(FETCH_NEW_KNOWLEDGE_CHUNK, tenant_id, vector)
            
            knowledge_id_append = []
            chunk_append = []
            replace_ids = set()
            for result in result_chunk:
                knowledge_id = result["knowledge_id"]
                chunk_id = result["chunk_id"]
                content = result["content"]
                if chunk_id in chunk_ids:
                    continue
                
                if knowledge_id not in selected_knowledge_dict.keys():
                    selected_knowledge_dict[knowledge_id] = {"metadata": "", "chunk_ids": []}
                    knowledge_id_append.append(knowledge_id)
                else:
                    replace_ids.add(knowledge_id)
                
                selected_knowledge_dict[knowledge_id]["chunk_ids"].append(chunk_id)
                chunk_append.append({chunk_id: decrypt(content), "knowledge_id": knowledge_id})
                
            if len(knowledge_id_append) > 0:
                result_knowledge = await conn.fetch(FETCH_NEW_KNOWLEDGE, tenant_id, knowledge_id_append)
                
                for result in result_knowledge:
                    knowledge_id = result["knowledge_id"]
                    metadata = result["metadata"]
                    selected_knowledge_dict[knowledge_id]["metadata"] = decrypt(metadata)
    
    except Exception as e:
        print(e)
        return "Failed fetch new knowledge from database."
    
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="Success fetch new knowledge from database.", 
                    tool_call_id=tool_call_id,
                    name="fetch_new_knowledge",
                )
            ],
            "selected_knowledge": {
                "append": [{k_id: val} for k_id, val in selected_knowledge_dict.items() if k_id in knowledge_id_append],
                "replace": [{k_id: val} for k_id, val in selected_knowledge_dict.items() if k_id in replace_ids],
            },
            "chunk_knowledge": {
                "append": chunk_append,
            },
        }
    )

## Tool: Fetch new memory (yang gak ada di state["memory_ids"])
@tool
async def fetch_new_memory(
    query: str, 
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command | str:
    """
    Fetch new memory entries for the user from the database that are not yet in the current state.

    Embeds the query, retrieves the top 5 closest memory entries via vector similarity search,
    filters out memories already present in state["memory_ids"], decrypts the content,
    and updates the state with the new memory entries and their IDs.

    Args:
        query (str): The search query to embed and use for similarity search.

    Returns:
        Command: Updates memory and memory_ids state on success.
        str: Error message on failure.
    """
    user_id = state["user_id"]
    tenant_id = state["tenant_id"]
    
    FETCH_NEW_MEMORY = """
    SELECT memory_id, content
    FROM User_memory
    WHERE user_id = $1 AND tenant_id = $2
    ORDER BY embedding <=> $3
    LIMIT 5;
    """
    
    vector = await gemini_embedding.aembed_query(query)
    
    try:
        async with pool.acquire() as conn:
            result_memory = await conn.fetch(FETCH_NEW_MEMORY, user_id, tenant_id, vector)
            
            memory_append = []
            memory_id_append = []
            for result in result_memory:
                memory_id = result["memory_id"]
                if memory_id not in state["memory_ids"]:
                    content = result["content"]
                    memory_append.append({memory_id: decrypt(content)})
                    memory_id_append.append(memory_id)
    
    except Exception as e:
        print(e)
        return "Failed fetch new memory from database."
    
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="Success fetch new memory from the database.",
                    tool_call_id=tool_call_id,
                    name="fetch_new_memory",
                )
            ],
            "memory_ids": {
                "append": memory_id_append,
            },
            "memory": {
                "append": memory_append,
            },
        }
    )

## Tool: Fetch new knowledge_session (yang gak ada di state["retrieved_session_knowledge"])
@tool
async def fetch_new_knowledge_session(
    query: str, 
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command | str:
    """
    Fetch new session knowledge chunks from the database that are not yet in the current state.

    Embeds the query, retrieves the top 5 closest session knowledge chunks via vector similarity
    search filtered by tenant and user, filters out chunks already present in the state,
    fetches and decrypts metadata for newly found session knowledge entries,
    and updates the state with the new session knowledge and chunks.

    Args:
        query (str): The search query to embed and use for similarity search.

    Returns:
        Command: Updates retrieved_session_knowledge and chunk_retrieved_session_knowledge state on success.
        str: Error message on failure.
    """
    tenant_id = state["tenant_id"]
    user_id = state["user_id"]
    chunk_ids = [c_id for k in state["chunk_retrieved_session_knowledge"] for k_id in k for c_id in k[k_id]["chunk_ids"]]
    selected_knowledge_dict = {k_id: val for item in state["chunk_retrieved_session_knowledge"] for k_id, val in item.items()}
    
    FETCH_NEW_KNOWLEDGE_CHUNK = """
    SELECT chunk_id, s_knowledge_id, content
    FROM Session_vectors
    WHERE tenant_id = $1 AND user_id = $2
    ORDER BY embedding <=> $3
    LIMIT 5;
    """
    
    FETCH_NEW_KNOWLEDGE = """
    SELECT s_knowledge_id, metadata
    FROM Session_knowledges
    WHERE tenant_id = $1 AND s_knowledge_id = ANY($2) AND user_id = $3;
    """
    
    vector = await gemini_embedding.aembed_query(query)
    
    try:
        async with pool.acquire() as conn:
            result_chunk = await conn.fetch(FETCH_NEW_KNOWLEDGE_CHUNK, tenant_id, user_id, vector)
            
            knowledge_id_append = []
            chunk_append = []
            replace_ids = set()
            for result in result_chunk:
                knowledge_id = result["s_knowledge_id"]
                chunk_id = result["chunk_id"]
                content = result["content"]
                if chunk_id in chunk_ids:
                    continue
                    
                if knowledge_id not in selected_knowledge_dict.keys():
                    selected_knowledge_dict[knowledge_id] = {"metadata": "", "chunk_ids": []}
                    knowledge_id_append.append(knowledge_id)
                else:
                    replace_ids.add(knowledge_id)
                    
                selected_knowledge_dict[knowledge_id]["chunk_ids"].append(chunk_id)
                chunk_append.append({chunk_id: decrypt(content), "knowledge_id": knowledge_id})
                
            if len(knowledge_id_append) > 0:
                result_knowledge = await conn.fetch(FETCH_NEW_KNOWLEDGE, tenant_id, knowledge_id_append, user_id)
                
                for result in result_knowledge:
                    knowledge_id = result["s_knowledge_id"]
                    metadata = result["metadata"]
                    selected_knowledge_dict[knowledge_id]["metadata"] = decrypt(metadata)
                    
    except Exception as e:
        print(e)
        return "Failed fetch new session knowledge from database."
        
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="Success fetch new session knowledge from the database.",
                    tool_call_id=tool_call_id,
                    name="fetch_new_knowledge_session",
                )
            ],
            "retrieved_session_knowledge": {
                "append": [{k_id: val} for k_id, val in selected_knowledge_dict.items() if k_id in knowledge_id_append],
                "replace": [{k_id: val} for k_id, val in selected_knowledge_dict.items() if k_id in replace_ids],
            },
            "chunk_retrieved_session_knowledge": {
                "append": chunk_append,
            },
        }
    )
    
## Tool: calculator, etc
@tool
async def calculator():
    """Use calculator to count numbers"""
    pass

## Tool: web search
@tool
async def web_search():
    """Use web search to find some external information"""
    pass

## Define Tools node
tools = [put_new_memory, fetch_new_knowledge, fetch_new_memory, fetch_new_knowledge_session, calculator, web_search]

# llm_instruct = llm_instruct.bind_tools(tools)
# llm_thinking = llm_thinking.bind_tools(tools)
# llm_coding = llm_coding.bind_tools(tools)

# gemini = gemini.bind_tools(tools)
gemini_instruct_tools = gemini_instruct.bind_tools(tools)
gemini_thinking_reasoning_tools = gemini_thinking_reasoning.bind_tools(tools)

tools_by_name = {tool.name: tool for tool in tools}

async def call_tools(state: State):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = await tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
        
        if not isinstance(tool_result, Command):
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
    return {}

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
    return {}

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
    return {}

## RAG (retrieve data from database based on just new query)
async def extract_content(message) -> str:
    content = message.content

    if isinstance(content, list): # AI / AI tool call
        content = " ".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        
    if isinstance(message, AIMessage) and message.tool_calls:
        tool_names = [t["name"] for t in messag.tool_calls]
        return f"AI: content: {content}, tool calls: {tool_names}"
    
    return content
    
async def trimming_message(messages):
    trimmed_msg = []
    for msg in messages:
        if msg.type == "tool":
            trimmed_msg.append({"role": msg.type, "content": msg.content})
        else: # human or ai
            trimmed_msg.append({"role": msg.type, "content": await extract_content(msg)})
            
    return trimmed_msg

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
    
    trimmed_msg = await trimming_message(messages)
    
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
    {trimmed_msg[:-1]}
    
    Latest message: {trimmed_msg[-1]}
    
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
    return {}

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
    
    trimmed_msg = await trimming_message(messages)
    
    if state["mode"] == "auto":
        system_query = f"""
        You are a routing agent. Your job is to classify the latest user message into exactly one route.

        Respond with ONLY the route name. No explanation, no punctuation — just the route name.

        Available routes:
        - basic: General questions, casual conversation, factual lookups, summarization, translation, or simple writing tasks. No coding involved.
        - coding_basic: Coding questions that are straightforward — syntax help, simple scripts, explaining code, fixing minor bugs, or short code generation.
        - coding_react: (reasoning-action) Coding tasks that require deeper analysis — architecture design, code review, debugging complex issues, performance optimization, building systems, or multi-step implementation.
        - thinking_react: (reasoning-action) Non-coding tasks that require deep reasoning — essay writing, evaluation, critical analysis, planning, decision-making, or multi-step problem solving.

        Rules:
        - If the task involves code → choose coding_basic or coding_react (never basic or thinking_react).
        - If unsure between coding_basic and coding_react → prefer coding_react.
        - If unsure between basic and thinking_react → prefer thinking_react.
        - Base your decision on the latest message. Use history only to resolve ambiguity (e.g., if the latest message refers to a previous topic).

        History messages:
        {trimmed_msg[:-1]}

        Latest message:
        {trimmed_msg[-1]}
        """
        
    elif state["mode"] == "thinking":
        system_query = f"""
        You are a routing agent. Your job is to classify the latest user message into exactly one route.

        Respond with ONLY the route name. No explanation, no punctuation — just the route name.

        Available routes:
        - coding_react: (reasoning-action) Coding tasks that require deeper analysis — architecture design, code review, debugging complex issues, performance optimization, building systems, or multi-step implementation.
        - thinking_react: (reasoning-action) Non-coding tasks that require deep reasoning — essay writing, evaluation, critical analysis, planning, decision-making, or multi-step problem solving.

        Rules:
        - If the task involves code → choose coding_react (never thinking_react).
        - Base your decision on the latest message. Use history only to resolve ambiguity (e.g., if the latest message refers to a previous topic).

        History messages:
        {trimmed_msg[:-1]}

        Latest message:
        {trimmed_msg[-1]}
        """
        
    else: # fast
        system_query = f"""
        You are a routing agent. Your job is to classify the latest user message into exactly one route.

        Respond with ONLY the route name. No explanation, no punctuation — just the route name.

        Available routes:
        - basic: General questions, casual conversation, factual lookups, summarization, translation, or simple writing tasks. No coding involved.
        - coding_basic: Coding questions that are straightforward — syntax help, simple scripts, explaining code, fixing minor bugs, or short code generation.

        Rules:
        - If the task involves code → choose coding_basic (never basic).
        - Base your decision on the latest message. Use history only to resolve ambiguity (e.g., if the latest message refers to a previous topic).

        History messages:
        {trimmed_msg[:-1]}

        Latest message:
        {trimmed_msg[-1]}
        """
        
    response = router_model.ainvoke(SystemMessage(content=system_query))
    
    if response["route"] == "coding_react" or response["route"] == "thinking_react":
        return Command(
            goto="reasoning",
            update={"route": response["route"]}
        )
    elif response["route"] == "coding_basic":
        return Command(
            goto="coding_basic",
            update={"route": response["route"]}
        )
    else: # basic
        return Command(
            goto="basic",
            update={"route": response["route"]}
        )
    
## Agent: Basic (same as router model), Visual (Photo, Video) Analysis (non-thinking)
async def basic(state: State):
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=gemini_instruct,
        max_tokens=8000,
        start_on="human",
        end_on=("human","tool"),
        include_system=True
    )
    
    metadata_knowledge = {k_id: val["metadata"] for x in state["selected_knowledge"] for k_id, val in x.items()}
    metadata_s_knowledge = {k_id: val["metadata"] for x in state["retrieved_session_knowledge"] for k_id, val in x.items()}
    
    knowledges = [{"chunk": chunk, "metadata": metadata_knowledge[k_id]} for x in state["chunk_knowledge"] for chunk, k_id in [x.values()]]
    s_knowledges = [{"chunk": chunk, "metadata": metadata_s_knowledge[k_id]} for x in state["chunk_retrieved_session_knowledge"] for chunk, k_id in [x.values()]]
    
    memories = [val for x in state["memory"] for _, val in x]
    
    system_query = f"""
    You are a helpful, concise, and friendly assistant.

    Answer the user's latest message clearly and directly based on the conversation history.

    You have been provided with the following context — use them as your primary reference before considering any tool calls:

    Knowledge from the tenant admin (general reference provided by the system):
    {knowledges}

    Knowledge uploaded by the user (session-specific reference):
    {s_knowledges}

    User memory (personal preferences and past context of the user):
    {memories}

    You have access to the following tools. Use them ONLY when the provided context above is insufficient or missing:
    - fetch_new_knowledge: Fetch additional knowledge from the database. Use only if the topic is not covered in the knowledge provided above.
    - fetch_new_knowledge_session: Fetch additional session-specific knowledge. Use only if the user-uploaded knowledge above is missing and clearly needed.
    - fetch_new_memory: Fetch additional user memory. Use only if the user memory above is insufficient to answer well.
    - put_new_memory: Save important new information about the user. Use only when the user explicitly shares personal preferences, goals, or facts worth remembering.
    - web_search: Search the web for external or real-time information. Use only when the answer cannot be found in the provided context or your own knowledge.
    - calculator: Perform numerical calculations. Use only when precise computation is needed.

    Keep your response concise and on point.
    """
    
    final_query = [SystemMessage(content=system_query), *messages]
    
    if state["streaming_mode"] == True:
        pass
    else:
        response = gemini_instruct_tools.ainvoke(final_query)
    
    return {"messages": response}

## Agent: Coding basic
async def coding_basic(state: State):
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=gemini_instruct,
        max_tokens=8000,
        start_on="human",
        end_on=("human","tool"),
        include_system=True
    )
    
    metadata_knowledge = {k_id: val["metadata"] for x in state["selected_knowledge"] for k_id, val in x.items()}
    metadata_s_knowledge = {k_id: val["metadata"] for x in state["retrieved_session_knowledge"] for k_id, val in x.items()}
    
    knowledges = [{"chunk": chunk, "metadata": metadata_knowledge[k_id]} for x in state["chunk_knowledge"] for chunk, k_id in [x.values()]]
    s_knowledges = [{"chunk": chunk, "metadata": metadata_s_knowledge[k_id]} for x in state["chunk_retrieved_session_knowledge"] for chunk, k_id in [x.values()]]
    
    memories = [val for x in state["memory"] for _, val in x]
    
    system_query = f"""
    You are a helpful and concise coding assistant.

    Answer the user's latest coding question clearly and directly based on the conversation history. Provide clean, working code with brief explanation when needed.

    You have been provided with the following context — use them as your primary reference before considering any tool calls:

    Knowledge from the tenant admin (general reference provided by the system):
    {knowledges}

    Knowledge uploaded by the user (session-specific reference):
    {s_knowledges}

    User memory (personal preferences and past context of the user, e.g., preferred language, framework, or coding style):
    {memories}

    You have access to the following tools. Use them ONLY when the provided context above is insufficient or missing:
    - fetch_new_knowledge: Fetch additional knowledge from the database. Use only if the topic or codebase context is not covered in the knowledge provided above.
    - fetch_new_knowledge_session: Fetch additional session-specific knowledge. Use only if the user-uploaded knowledge above is missing and clearly needed.
    - fetch_new_memory: Fetch additional user memory. Use only if the user memory above is insufficient to answer well.
    - put_new_memory: Save important new information about the user. Use only when the user explicitly shares coding preferences or setup worth remembering.
    - web_search: Search for external information such as library docs, changelogs, or error references. Use only when the answer is not available in the provided context or your own knowledge.
    - calculator: Perform numerical calculations. Use only when precise computation is needed.

    Keep your response concise. Avoid over-engineering — match the complexity of your answer to the simplicity of the question.
    """
    
    final_query = [SystemMessage(content=system_query), *messages]
    
    if state["streaming_mode"] == True:
        pass
    else:
        response = gemini_instruct_tools.ainvoke(final_query)
    
    return {"messages": response}
    
## Agent: Coding react
async def coding_react(state: State):
    last_thought = state["messages"][-1]
    
    metadata_knowledge = {k_id: val["metadata"] for x in state["selected_knowledge"] for k_id, val in x.items()}
    metadata_s_knowledge = {k_id: val["metadata"] for x in state["retrieved_session_knowledge"] for k_id, val in x.items()}
    
    knowledges = [{"chunk": chunk, "metadata": metadata_knowledge[k_id]} for x in state["chunk_knowledge"] for chunk, k_id in [x.values()]]
    s_knowledges = [{"chunk": chunk, "metadata": metadata_s_knowledge[k_id]} for x in state["chunk_retrieved_session_knowledge"] for chunk, k_id in [x.values()]]
    
    memories = [val for x in state["memory"] for _, val in x]
    
    if isinstance(last_thought, ToolMessage):
        messages = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=gemini_instruct,
            max_tokens=8000,
            start_on="human",
            end_on=("human","tool"),
            include_system=True
        )
        
        trimmed_msg = await trimming_message(messages)
        
        msg = state["last_query"]
        
        system_prompt = f"""
        You are a precise coding assistant operating in a reasoning-action workflow.

        You have just received results from one or more tool calls. Based on the conversation history below — including previous reasoning steps, actions, and tool results — synthesize the findings and answer the original query accurately.

        You have also been provided with the following context as additional reference:

        Knowledge from the tenant admin (general reference provided by the system):
        {knowledges}

        Knowledge uploaded by the user (session-specific reference):
        {s_knowledges}

        User memory (personal preferences and past context of the user, e.g., preferred language, framework, or coding style):
        {memories}

        Original query:
        {msg}

        Conversation history (including reasoning, actions, and tool results):
        {trimmed_msg}

        Instructions:
        - Summarize the relevant tool result(s) briefly.
        - Use the summarized findings and provided context to directly answer the original query.
        - If the tool results and provided context are insufficient to fully answer the query, you may call additional tools — but ONLY if strictly necessary.
        - Do not repeat information already established in the history.
        - Keep your final answer technically accurate and concise.
        """
        
    else:
        msg = last_thought.content[0]["text"].replace("Thought: QUERY:", "").strip()
        
        system_prompt = f"""
        You are a precise coding assistant operating in a reasoning-action workflow.

        Answer the following query as accurately and concisely as possible:
        {msg}

        You have been provided with the following context — use them as your primary reference before considering any tool calls:

        Knowledge from the tenant admin (general reference provided by the system):
        {knowledges}

        Knowledge uploaded by the user (session-specific reference):
        {s_knowledges}

        User memory (personal preferences and past context of the user, e.g., preferred language, framework, or coding style):
        {memories}

        You have access to the following tools. Use them ONLY if the query cannot be answered from the provided context or your existing knowledge:
        - fetch_new_knowledge: Fetch additional knowledge from the database. Use only if codebase or technical context is not covered in the knowledge provided above.
        - fetch_new_knowledge_session: Fetch additional session-specific knowledge. Use only if session-level code context is missing and clearly needed.
        - fetch_new_memory: Fetch additional user memory. Use only if the user memory above is insufficient.
        - put_new_memory: Save important user information. Use only when explicitly needed.
        - web_search: Search for library docs, API references, changelogs, or error explanations. Use only when the answer is not available in the provided context or your knowledge.
        - calculator: Perform precise numerical computation when needed.

        Return a focused, technically accurate answer. Do not over-explain.
        """
    
    result = await gemini_thinking_reasoning_tools.ainvoke(SystemMessage(content=system_prompt))
    
    if result.tool_calls:
        return {
            "messages": [AIMessage(content=f"Observation with tools: {extract_content(result)}")],
            "last_query": msg,
        }
    else:
        return {
            "messages": [
                AIMessage(content=f"Action: query('{msg}')"), 
                AIMessage(content=f"Observation: {result.content[0]['text']}")
            ],
            "last_query": msg,
            "iteration": state.get("iteration", 0) + 1,
            "reasoning_questions_obervation": [{"question": msg, "observation": result.content[0]['text']}],
        }
    
## Agent: Coding end (conclusion)
async def coding_end(state: State):
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=gemini_instruct,
        max_tokens=8000,
        start_on="human",
        end_on=("human","tool"),
        include_system=True
    )
    
    metadata_knowledge = {k_id: val["metadata"] for x in state["selected_knowledge"] for k_id, val in x.items()}
    metadata_s_knowledge = {k_id: val["metadata"] for x in state["retrieved_session_knowledge"] for k_id, val in x.items()}
    
    knowledges = [{"chunk": chunk, "metadata": metadata_knowledge[k_id]} for x in state["chunk_knowledge"] for chunk, k_id in [x.values()]]
    s_knowledges = [{"chunk": chunk, "metadata": metadata_s_knowledge[k_id]} for x in state["chunk_retrieved_session_knowledge"] for chunk, k_id in [x.values()]]
    
    memories = [val for x in state["memory"] for _, val in x]
    
    system_prompt = f"""
    You are a precise coding assistant.

    The reasoning-action workflow has completed. You have also been provided with the following context as additional reference:

    Knowledge from the tenant admin (general reference provided by the system):
    {knowledges}

    Knowledge uploaded by the user (session-specific reference):
    {s_knowledges}

    User memory (personal preferences and past context of the user, e.g., preferred language, framework, or coding style):
    {memories}

    Below are the reasoning questions and their corresponding observations gathered throughout the process:
    {state["reasoning_questions_observation"]}

    Your task:
    - Synthesize the questions, observations, and provided context into a single, coherent, and complete answer.
    - Do not omit any key findings or conclusions from the reasoning process.
    - Present code snippets, explanations, or technical details in a clean and structured format.
    - Use the conversation history for context if needed to align your answer with the user's original intent.
    """
    
    if state["streaming_mode"] == True:
        pass
    else:
        response = await gemini_instruct.ainvoke([SystemMessage(content=system_prompt), *messages])
        
    return response
    
## Agent: Thinking react
async def thinking_react(state: State):
    last_thought = state["messages"][-1]
    
    metadata_knowledge = {k_id: val["metadata"] for x in state["selected_knowledge"] for k_id, val in x.items()}
    metadata_s_knowledge = {k_id: val["metadata"] for x in state["retrieved_session_knowledge"] for k_id, val in x.items()}
    
    knowledges = [{"chunk": chunk, "metadata": metadata_knowledge[k_id]} for x in state["chunk_knowledge"] for chunk, k_id in [x.values()]]
    s_knowledges = [{"chunk": chunk, "metadata": metadata_s_knowledge[k_id]} for x in state["chunk_retrieved_session_knowledge"] for chunk, k_id in [x.values()]]
    
    memories = [val for x in state["memory"] for _, val in x]
    
    if isinstance(last_thought, ToolMessage):
        messages = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=gemini_instruct,
            max_tokens=8000,
            start_on="human",
            end_on=("human","tool"),
            include_system=True
        )
        
        trimmed_msg = await trimming_message(messages)
        
        msg = state["last_query"]
        
        system_prompt = f"""
        You are a precise reasoning assistant operating in a reasoning-action workflow.

        You have just received results from one or more tool calls. Based on the conversation history below — including previous reasoning steps, actions, and tool results — synthesize the findings and answer the original query accurately.

        You have also been provided with the following context as additional reference:

        Knowledge from the tenant admin (general reference provided by the system):
        {knowledges}

        Knowledge uploaded by the user (session-specific reference):
        {s_knowledges}

        User memory (personal preferences and past context of the user):
        {memories}

        Original query:
        {msg}

        Conversation history (including reasoning, actions, and tool results):
        {trimmed_msg}

        Instructions:
        - Summarize the relevant tool result(s) briefly.
        - Use the summarized findings and provided context to directly answer the original query.
        - If the tool results and provided context are insufficient to fully answer the query, you may call additional tools — but ONLY if strictly necessary.
        - Do not repeat information already established in the history.
        - Keep your final answer well-reasoned, clear, and to the point.
        """
        
    else:
        msg = last_thought.content[0]["text"].replace("Thought: QUERY:", "").strip()
        
        system_prompt = f"""
        You are a precise reasoning assistant operating in a reasoning-action workflow.

        Answer the following query as accurately and concisely as possible:
        {msg}

        You have been provided with the following context — use them as your primary reference before considering any tool calls:

        Knowledge from the tenant admin (general reference provided by the system):
        {knowledges}

        Knowledge uploaded by the user (session-specific reference):
        {s_knowledges}

        User memory (personal preferences and past context of the user):
        {memories}

        You have access to the following tools. Use them ONLY if the query cannot be answered from the provided context or your existing knowledge:
        - fetch_new_knowledge: Fetch additional knowledge from the database. Use only if the topic context is not covered in the knowledge provided above.
        - fetch_new_knowledge_session: Fetch additional session-specific knowledge. Use only if session context is missing and clearly needed.
        - fetch_new_memory: Fetch additional user memory. Use only if the user memory above is insufficient.
        - put_new_memory: Save important user information. Use only when explicitly needed.
        - web_search: Search for external, real-time, or reference information. Use only when the answer is not available in the provided context or your knowledge.
        - calculator: Perform precise numerical computation when needed.

        Return a focused, well-reasoned answer. Be direct — avoid unnecessary elaboration.
        """
    
    result = await gemini_thinking_reasoning_tools.ainvoke(SystemMessage(content=system_prompt))
    
    if result.tool_calls:
        return {
            "messages": [AIMessage(content=f"Observation with tools: {extract_content(result)}")],
            "last_query": msg,
        }
    else:
        return {
            "messages": [
                AIMessage(content=f"Action: query('{msg}')"), 
                AIMessage(content=f"Observation: {result.content[0]['text']}")
            ],
            "last_query": msg,
            "iteration": state.get("iteration", 0) + 1,
            "reasoning_questions_obervation": [{"question": msg, "observation": result.content[0]['text']}],
        }
    
## Agent: Thinking end (conclusion)
async def thinking_end(state: State):
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=gemini_instruct,
        max_tokens=8000,
        start_on="human",
        end_on=("human","tool"),
        include_system=True
    )
    
    metadata_knowledge = {k_id: val["metadata"] for x in state["selected_knowledge"] for k_id, val in x.items()}
    metadata_s_knowledge = {k_id: val["metadata"] for x in state["retrieved_session_knowledge"] for k_id, val in x.items()}
    
    knowledges = [{"chunk": chunk, "metadata": metadata_knowledge[k_id]} for x in state["chunk_knowledge"] for chunk, k_id in [x.values()]]
    s_knowledges = [{"chunk": chunk, "metadata": metadata_s_knowledge[k_id]} for x in state["chunk_retrieved_session_knowledge"] for chunk, k_id in [x.values()]]
    
    memories = [val for x in state["memory"] for _, val in x]
    
    system_prompt = f"""
    You are a precise reasoning assistant.

    The reasoning-action workflow has completed. You have also been provided with the following context as additional reference:

    Knowledge from the tenant admin (general reference provided by the system):
    {knowledges}

    Knowledge uploaded by the user (session-specific reference):
    {s_knowledges}

    User memory (personal preferences and past context of the user):
    {memories}

    Below are the reasoning questions and their corresponding observations gathered throughout the process:
    {state["reasoning_questions_observation"]}

    Your task:
    - Synthesize the questions, observations, and provided context into a single, coherent, and complete answer.
    - Do not omit any key findings, insights, or conclusions from the reasoning process.
    - Present your answer in a clear, well-structured, and readable format.
    - Use the conversation history for context if needed to align your answer with the user's original intent.
    """
    
    if state["streaming_mode"] == True:
        pass
    else:
        response = await gemini_instruct.ainvoke([SystemMessage(content=system_prompt), *messages])
        
    return response
    
## Reasoning node untuk Agent Thinking dan Coding (untuk pengembangan: tambahkan interrupt dan/atau user input sebelum masuk reasoning node)
async def reasoning(state: State):
    # route = state["route"] # coding_react or thinking_react
    
    if state["route"] == "coding_react":
        node_end = "coding_end"
    else: # thinking_react
        node_end = "thinking_end"
    
    iteration = state.get("iteration", 0)
    if iteration >= 3:
        return Command(
            goto=node_end,
            update={
                "messages": AIMessage(content="Thought: I have gathered enough information"),
                "iteration": iteration,
            }
        )
        
    metadata_knowledge = {k_id: val["metadata"] for x in state["selected_knowledge"] for k_id, val in x.items()}
    metadata_s_knowledge = {k_id: val["metadata"] for x in state["retrieved_session_knowledge"] for k_id, val in x.items()}
    
    knowledges = [{"chunk": chunk, "metadata": metadata_knowledge[k_id]} for x in state["chunk_knowledge"] for chunk, k_id in [x.values()]]
    s_knowledges = [{"chunk": chunk, "metadata": metadata_s_knowledge[k_id]} for x in state["chunk_retrieved_session_knowledge"] for chunk, k_id in [x.values()]]
    
    memories = [val for x in state["memory"] for _, val in x]
            
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=gemini_thinking_reasoning,
        max_tokens=8000,
        start_on="human",
        end_on=("human","tool"),
        include_system=True
    )
    
    trimmed_msg = await trimming_message(messages)
    
    prompt = f"""
    You are a precise reasoning agent tasked with breaking down a complex query into focused sub-queries to gather the necessary information.

    You have been provided with the following context as an initial reference:

    Knowledge from the tenant admin (general reference provided by the system):
    {knowledges}

    Knowledge uploaded by the user (session-specific reference):
    {s_knowledges}

    User memory (personal preferences and past context of the user):
    {memories}

    Original query:
    {trimmed_msg[-1]}

    Reasoning history so far (previous queries and observations):
    {trimmed_msg[:-1]}

    Queries completed: {iteration}/3

    You MUST generate exactly 3 queries in total to fully gather the information needed to answer the original query.
    Each query must target a specific piece of information not yet covered by the provided context or previous observations.

    Respond ONLY with:
    QUERY: <your specific question>

    Do NOT be conversational. Do NOT explain your reasoning. Do NOT thank the user. Output ONLY: QUERY: <question>
    """
    
    decision = gemini_thinking_reasoning.ainvoke(SystemMessage(content=prompt))
    
    if decision.content[0]["text"].startswith("QUERY:"):
        return Command(
            goto=state["route"],
            update={
                "messages": AIMessage(content=f"Thought: {decision.content[0]['text']}"),
                "iteration": iteration,
            }
        )
    
    return Command(
        goto=node_end,
        update={
            "messages": AIMessage(content=f"Thought: {decision.content[0]['text']}"),
            "iteration": iteration,
        }
    )
 
## Agent: Report Maker (Orchestration)

## Agent: 


# Define agent
async def get_agent():
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
    builder.add_node("coding_react", coding_react)
    builder.add_node("coding_end", coding_end)
    builder.add_node("thinking_react", thinking_react)
    builder.add_node("thinking_end", thinking_end)
    builder.add_node("reasoning", reasoning)
    builder.add_node("call_tools", call_tools)
    
    
    builder.add_edge(START, "check_knowledge_session_ttl")
    builder.add_edge("check_knowledge_session_ttl", "fetch_knowledge_session")
    builder.add_edge("check_knowledge_session_ttl", "check_knowledge_exist")
    builder.add_edge("fetch_knowledge_session", "judge_knowledge_session")
    builder.add_edge("judge_knowledge_session", "check_knowledge_exist")

    builder.add_edge("check_knowledge_exist", "fetch_knowledge")
    builder.add_edge("check_knowledge_exist", "check_memory_exist_and_fetch")
    builder.add_edge("fetch_knowledge", "judge_knowledge")
    builder.add_edge("judge_knowledge", "check_memory_exist_and_fetch")
    
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
    builder.add_edge("router", "basic")
    builder.add_edge("router", "coding_basic")
    builder.add_edge("router", "reasoning")
    # builder.add_conditional_edges("router", lambda s: s["route"], ["basic", "coding_basic", "reasoning"])
    
    # non-reasoning (basic and coding)
    builder.add_conditional_edges("basic", should_continue, ["call_tools", END])
    builder.add_edge("call_tools", "basic")
    
    builder.add_conditional_edges("coding_basic", should_continue, ["call_tools", END])
    builder.add_edge("call_tools", "coding_basic")
    
    # thinking-reasoning (coding and thinking)
    builder.add_edge("reasoning", "coding_react")
    builder.add_edge("reasoning", "coding_end")
    builder.add_edge("reasoning", "thinking_react")
    builder.add_edge("reasoning", "thinking_end")
    # builder.add_conditional_edges("reasoning", lambda s: s["route"], ["coding_react", "coding_end", "thinking_react", "thinking_end"])
    ## coding
    builder.add_conditional_edges("coding_react", should_continue, ["call_tools", "reasoning"])
    builder.add_edge("call_tools", "coding_react")
    ## thinking
    builder.add_conditional_edges("thinking_react", should_continue, ["call_tools", "reasoning"])
    builder.add_edge("call_tools", "thinking_react")
    
    return builder
    
async def streaming(db_pool, input_data: dict, f: Optional[UploadFile] = None):
    global pool
    pool = db_pool
    
    builder = await get_agent()
    
    thread_id = input_data["thread_id"]
    user_id = input_data["user_id"]
    tenant_id = input_data["tenant_id"]
    input_prompt = input_data["input_prompt"]
    mode = input_data["mode"]
    
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
                {"messages": [HumanMessage(content=input_prompt)], "mode": mode, "streaming_mode": True},
                config,
                stream_mode="custom",
                version="v2",
            ):
                if chunk["type"] == "messages":
                    msg, metadata = chunk["data"]
                    if msg.content:
                        yield msg.content #chunk["messages"][-1]

async def chat(db_pool, input_data: dict, f: Optional[UploadFile] = None):
    global pool
    pool = db_pool
    
    builder = await get_agent()
    
    thread_id = input_data["thread_id"]
    user_id = input_data["user_id"]
    tenant_id = input_data["tenant_id"]
    input_prompt = input_data["input_prompt"]
    mode = input_data["mode"]
    
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
                
        result = agent.ainvoke(
            {"messages": [HumanMessage(content=input_prompt)], "mode": mode, "streaming_mode": False},
            config,
        )
        
    return message_to_dict(result["message"][-1])

async def get_agent_graph():
    builder = await get_agent()
    
    agent = builder.compile()
    
    png_graph = agent.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_graph)

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
    
    vector = await gemini_embedding.aembed_documents(chunks)
    # vector_prompt = await gemini_embedding.aembed_query(prompt)

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
    
    vector = await gemini_embedding.aembed_documents(chunks)

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

if __name__ == "__main__":
    asyncio.run(get_agent_graph())


# For next development
# Use cron job to update state of the retrieved document ids. delete the id that will have been remove in the State
# Paralellize retrive knowledge/memory
# pisahkan state antara messages ke user dan system messages (ai/tool)
# Menggunakan model khusus coding dan bukan

# react ref: https://machinelearningmastery.com/building-react-agents-with-langgraph-a-beginners-guide/