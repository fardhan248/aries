from models.gemini import gemini, gemini_instruct, gemini_thinking_reasoning, gemini_embedding

from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage
from langchain_core.messages.utils import trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langgraph.prebuilt import InjectedState 
from langchain_core.tools import InjectedToolCallId
from langchain_community.tools import DuckDuckGoSearchResults

from typing_extensions import Annotated, Any
import asyncio, fitz, uuid, base64, copy, numexpr, json, httpx
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from utils.documents_utils import encrypt, decrypt
from src.states import State, RouterOutput, CalculatorExpression

from string_utils.database_queries import Queries
from string_utils.prompts import Prompts

queries = Queries()
prompts = Prompts()

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
    
    try:
        async with pool.acquire() as conn:
            memories = await conn.fetch(queries.CHECK_MEMORY, user_id, tenant_id)
            
            if len(memories) >= 20:
                return "Failed put new memory to the database. The user's saved memory has reached limit memory: 20" 
                
            encrypted_query = await encrypt(query)
            await conn.execute(queries.PUT_NEW_MEMORY, memory_id, tenant_id, user_id, encrypted_query, vector[0])
            
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
    
    vector = await gemini_embedding.aembed_query(query)
    
    try:
        async with pool.acquire() as conn:
            result_chunk = await conn.fetch(queries.FETCH_NEW_KNOWLEDGE_CHUNK, tenant_id, vector)
            
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
                
                decrypted_content = await decrypt(content)
                selected_knowledge_dict[knowledge_id]["chunk_ids"].append(chunk_id)
                chunk_append.append({chunk_id: decrypted_content, "knowledge_id": knowledge_id})
                
            if len(knowledge_id_append) > 0:
                result_knowledge = await conn.fetch(queries.FETCH_NEW_KNOWLEDGE, tenant_id, knowledge_id_append)
                
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

    vector = await gemini_embedding.aembed_query(query)
    
    try:
        async with pool.acquire() as conn:
            result_memory = await conn.fetch(queries.FETCH_NEW_MEMORY, user_id, tenant_id, vector)
            
            memory_append = []
            memory_id_append = []
            for result in result_memory:
                memory_id = result["memory_id"]
                if memory_id not in state["memory_ids"]:
                    content = await decrypt(result["content"])
                    memory_append.append({memory_id: content})
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
    thread_id = state["thread_id"]
    chunk_ids = [c_id for k in state["chunk_retrieved_session_knowledge"] for k_id in k for c_id in k[k_id]["chunk_ids"]]
    selected_knowledge_dict = {k_id: val for item in state["chunk_retrieved_session_knowledge"] for k_id, val in item.items()}
    
    vector = await gemini_embedding.aembed_query(query)
    
    try:
        async with pool.acquire() as conn:
            result_chunk = await conn.fetch(queries.FETCH_NEW_KNOWLEDGE_SESSION_CHUNK, tenant_id, user_id, thread_id, vector)
            
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
                    
                decrypted_content = await decrypt(content)
                selected_knowledge_dict[knowledge_id]["chunk_ids"].append(chunk_id)
                chunk_append.append({chunk_id: decrypted_content, "knowledge_id": knowledge_id})
                
            if len(knowledge_id_append) > 0:
                result_knowledge = await conn.fetch(queries.FETCH_NEW_KNOWLEDGE_SESSION, tenant_id, knowledge_id_append, user_id, thread_id)
                
                for result in result_knowledge:
                    knowledge_id = result["s_knowledge_id"]
                    metadata = result["metadata"]
                    selected_knowledge_dict[knowledge_id]["metadata"] = await decrypt(metadata)
                    
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
async def calculator(expression: CalculatorExpression) -> str:
    """
    Evaluate a mathematical expression using numexpr.

    The expression must contain:
    - "expr": a mathematical expression string. Variables that require numpy values
              (e.g. np.pi, np.e, np.mean) must be written as placeholders e.g. "{var1}".
              Simple numeric literals can be written directly in the expression.
    - "vars": optional. Define ONLY if the expression contains variables that need
              numpy constants or computed numpy values (e.g. np.pi, np.e, np.inf,
              np.mean([...])). Omit or leave empty if all values are plain numbers.

    Supported operations:
    - Arithmetic     : +, -, *, /, //, **, %, <<, >>
    - Comparison     : <, <=, ==, !=, >=, >
    - Trigonometry   : sin, cos, tan, arcsin, arccos, arctan, arctan2, hypot
    - Exponential    : exp, expm1, exp2, log, log10, log1p, log2, sqrt
    - Rounding       : floor, ceil, abs
    - Conditional    : where(bool, val1, val2)
    - Utility        : isinf, isnan, isfinite, maximum, minimum

    Example (without vars):
        expression = {"expr": "3 ** 2 + sqrt(16) - 1.5"}
        result = await calculator(expression)
        # result: "11.5"

    Example (with vars, because needs numpy constant):
        expression = {
            "expr": "{a} ** 2 + sqrt({b}) - {pi}",
            "vars": {
                "a": 3.0,
                "b": 16.0,
                "pi": np.pi,
            }
        }
        result = await calculator(expression)

    Args:
        expression (CalculatorExpression): Pydantic model with fields:
            - expr (str): the mathematical expression.
            - vars (dict[str, Any]): optional variable mappings for numpy values.

    Returns:
        str: The result of the expression, or an error message if evaluation fails.
    """
    try:
        expr = expression.expr.format_map(expression.variables)
        result = numexpr.evaluate(expr)
        return str(result.item() if hasattr(result, "item") else result)
    except Exception as e:
        print(e)
        return f"Failed to evaluate expression: {expression}."

## Tool: web search
search = DuckDuckGoSearchResults(output_format="list")

@tool
async def web_search(query: str) -> list[dict[str, str]]:
    """
    Search the web using DuckDuckGo and return a list of relevant results.

    Use this tool ONLY when the required information is not available in the current
    context, knowledge, or memory — such as real-time data, recent events, or
    external references.

    Args:
        query (str): The search query string.

    Returns:
        list[dict[str, str]]: A list of search results, where each result contains:
            - "snippet": brief description of the result.
            - "title"  : title of the page.
            - "link"   : URL of the page.
    """
    return await search.ainvoke(query)
    
    
# ## Tool: (geocoding) get latitude-longitude
# @tool
# async def get_latitude_longitude(location: str) -> tuple[float, float]:
    # """"""
    # try: 
        # async with httpx.AsyncClient() as client:
            # response = await client.get(
                # "https://geocoding-api.open-meteo.com/v1/search",
                # params={
                    # "name": location,
                    # "count": 1,
                # }
            # )
            
            # if response.status_code != 200:
                # return response.json()
            
            # result = response.json()["results"][0]
            
            # lat, lon = result["latitude"], result["longitude"]
            # return (lat, lon)
    
    # except Exception as e:
        # print(e)
        # return "Failed get lat-lon from API."

# ## Tool: weather forecast
# @tool
# async def get_weather_forecast():
    # return
    
# ## Tool: weather historical
# @tool
# async def get_weather_historical():
    # return

## Tool: get datetime now
@tool
async def get_datetime_now() -> str:
    """
    Get the current date and time in the Asia/Jakarta timezone (WIB, UTC+7).

    Use this tool when the user asks about the current time, date, or anything
    that requires knowing the present datetime.

    Returns:
        str: Current datetime string in the format:
             "Year-Month-Date Hour:Minute:Second: YYYY-MM-DD HH:MM:SS.ffffff+HH:MM"
    """
    return "Year-Month-Date Hour:Minute:Second: " + str(datetime.now(ZoneInfo("Asia/Jakarta")))

## Define Tools node
tools = [put_new_memory, fetch_new_knowledge, fetch_new_memory, fetch_new_knowledge_session, calculator, web_search, get_datetime_now]

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
async def check_knowledge_session_ttl(state: State):
    """If there is no chunk from database, drop knowledge indices"""
    retrieved_session_knowledge = copy.deepcopy(state["retrieved_session_knowledge"])
    
    if len(retrieved_session_knowledge) == 0:
        return Command(goto="check_knowledge_exist")
    
    tenant_id = state["tenant_id"]
    user_id = state["user_id"]
    thread_id = state["thread_id"]
    knowledge_ids = [list(s.keys())[0] for s in retrieved_session_knowledge]
    
    results = await pool.fetch(queries.QUERY_CHECK_KNOWLEDGE_SESSION_TTL, knowledge_ids, tenant_id, user_id, thread_id) # check if the document still exist in the database
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

async def fetch_knowledge_session(state: State, config: RunnableConfig):
    """
    Fetch knowledge session from existing retrieved_session_knowledge indexes.
    """    
    # Load chunk_retrieved_session_knowledge
    tenant_id = state["tenant_id"]
    user_id = state["user_id"]
    thread_id = state["thread_id"]
    chunk_ids = [c_id for k in state["retrieved_session_knowledge"] for k_id in k for c_id in k[k_id]["chunk_ids"]]
    
    chunk_knowledges = await pool.fetch(queries.QUERY_FETCH_KNOWLEDGE_SESSION, chunk_ids, tenant_id, user_id, thread_id)
    item_append = []
    for item in chunk_knowledges:
        s_knowledge_id = item["s_knowledge_id"]
        chunk_id = item["chunk_id"]
        content = await decrypt(item["content"])
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
async def check_knowledge_exist(state: State):
    """If there is no chunk from database, drop knowledge indices"""
    selected_knowledge = copy.deepcopy(state["selected_knowledge"])
    
    if len(selected_knowledge) == 0:
        return Command(goto="check_memory_exist")
    
    tenant_id = state["tenant_id"]
    knowledge_ids = [list(s.keys())[0] for s in selected_knowledge]
    
    results = await pool.fetch(queries.QUERY_CHECK_KNOWLEDGE_EXIST, knowledge_ids, tenant_id) # check if the document still exist in the database
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

async def fetch_knowledge(state: State):
    """
    Fetch knowledge from existing selected_knowledge indexes.
    """
    # Load chunk_knowledge
    tenant_id = state["tenant_id"]
    chunk_ids = [c_id for k in state["selected_knowledge"] for k_id in k for c_id in k[k_id]["chunk_ids"]]
    
    chunk_knowledges = await pool.fetch(queries.QUERY_FETCH_KNOWLEDGE, chunk_ids, tenant_id)
    item_append = []
    for item in chunk_knowledges:
        knowledge_id = item["knowledge_id"]
        chunk_id = item["chunk_id"]
        content = await decrypt(item["content"])
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
async def check_memory_exist_and_fetch(state: State):
    """If there is no chunk from database, drop memory indices"""
    tenant_id = state["tenant_id"]
    user_id = state["user_id"]
    memory_ids = copy.deepcopy(state["memory_ids"])
    
    if len(memory_ids) == 0:
        return Command(goto="rag")
    
    results = await pool.fetch(queries.QUERY_MEMORY_EXIST, memory_ids, tenant_id, user_id) # check if the memory still exist in the database
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
                content = await decrypt(item["content"])
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
       
    metadata_knowledge = {k_id: val["metadata"] for x in state["selected_knowledge"] for k_id, val in x.items()}    
    knowledges = [{"chunk": chunk, "metadata": metadata_knowledge[k_id]} for x in state["chunk_knowledge"] for chunk, k_id in [x.values()]]

    system_query = prompts.RAG_SYSTEM_QUERY.format_map({
        "trimmed_msg_rag": trimmed_msg[:-1],
        "latest_message": trimmed_msg[-1],
        "knowledges": knowledges,
    })
    
    llm_output = await gemini_instruct.ainvoke(SystemMessage(content=system_query))
    vector = await gemini_embedding.aembed_query(llm_output.content)
    
    # Retrieve from database 
    async with pool.acquire() as conn:
        rows_fetch_chunks = await conn.fetch(queries.FETCH_CHUNKS, tenant_id, vector)
        
        if len(rows_fetch_chunks) == 0:
            return Command(goto="router")
                
        knowledge_ids_from_chunks = list(set(x["knowledge_id"] for x in rows_fetch_chunks))
        
        rows_fetch_knowledges = await conn.fetch(queries.FETCH_KNOWLEDGE, tenant_id, knowledge_ids_from_chunks)
        
        knowledge_id_metadata = {k_id: x["metadata"] for k_id, x in zip(knowledge_ids_from_chunks, rows_fetch_knowledges)}
    
    # Check if the knowledge is already retrieved on chunk_knowledge
    item_append_knowledge = {k_id: val for x in selected_knowledge for k_id, val in x.items()}
    for k_id, meta in knowledge_id_metadata.items():
        if k_id not in knowledge_ids:
            metadata = await decrypt(meta)
            item_append_knowledge[k_id] = {"metadata": metadata, "chunk_ids": []}
    
    item_append = []
    ids_replace = set()
    for item in rows_fetch_chunks:
        chunk_id = item["chunk_id"]
        if chunk_id not in chunk_ids:
            content = await decrypt(item["content"])
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
    schema=RouterOutput.model_json_schema(), method="json_schema"
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
        system_query = prompts.AUTO_SYSTEM_QUERY.format_map({
            "trimmed_msg_auto": trimmed_msg[:-1],
            "latest_message": trimmed_msg[-1],
        })
        
    elif state["mode"] == "thinking":
        system_query = prompts.THINKING_SYSTEM_QUERY.format_map({
            "trimmed_msg_thinking": trimmed_msg[:-1],
            "latest_message": trimmed_msg[-1],
        })
        
    else: # fast
        system_query = prompts.FAST_SYSTEM_QUERY({
            "trimmed_msg_fast": trimmed_msg[:-1],
            "latest_message": trimmed_msg[-1],
        })
        
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
    
    system_query = prompts.BASIC_SYSTEM_QUERY.format_map({
        "knowledges": knowledges,
        "s_knowledges": s_knowledges,
        "memories": memories,
    })   
    
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
    
    system_query = prompts.CODING_BASIC_SYSTEM_QUERY.format_map({
        "knowledges": knowledges,
        "s_knowledges": s_knowledges,
        "memories": memories,
    })   
    
    final_query = [SystemMessage(content=system_query), *messages]
    
    if state["streaming_mode"] == True:
        pass
    else:
        response = gemini_instruct_tools.ainvoke(final_query)
    
    return {"messages": response}
    
## Agent: Coding react
async def coding_react(state: State):
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
    
    last_thought = state["messages"][-1]
    
    metadata_knowledge = {k_id: val["metadata"] for x in state["selected_knowledge"] for k_id, val in x.items()}
    metadata_s_knowledge = {k_id: val["metadata"] for x in state["retrieved_session_knowledge"] for k_id, val in x.items()}
    
    knowledges = [{"chunk": chunk, "metadata": metadata_knowledge[k_id]} for x in state["chunk_knowledge"] for chunk, k_id in [x.values()]]
    s_knowledges = [{"chunk": chunk, "metadata": metadata_s_knowledge[k_id]} for x in state["chunk_retrieved_session_knowledge"] for chunk, k_id in [x.values()]]
    
    memories = [val for x in state["memory"] for _, val in x]
    
    if isinstance(last_thought, ToolMessage):       
        msg = state["last_query"]
        
        system_prompt = prompts.CODING_REACT_TOOL_SYSTEM_QUERY.format_map({
            "knowledges": knowledges,
            "s_knowledges": s_knowledges,
            "memories": memories,
            "msg": msg,
            "trimmed_msg": trimmed_msg,
        })        
        
    else:
        msg = last_thought.content[0]["text"].replace("Thought: QUERY:", "").strip()
        
        system_prompt = prompts.CODING_REACT_SYSTEM_QUERY.format_map({
            "msg": msg,
            "knowledges": knowledges,
            "s_knowledges": s_knowledges,
            "memories": memories,
            "history": trimmed_msg[:-1],
        })    
    
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
            "reasoning_questions_observation": [{"question": msg, "observation": result.content[0]['text']}],
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
    
    system_prompt = prompts.CODING_END_SYSTEM_QUERY.format_map({
        "knowledges": knowledges,
        "s_knowledges": s_knowledges,
        "memories": memories,
        "reasoning_questions_observation": state["reasoning_questions_observation"],
    })
    
    if state["streaming_mode"] == True:
        pass
    else:
        response = await gemini_instruct.ainvoke([SystemMessage(content=system_prompt), *messages])
        
    return response
    
## Agent: Thinking react
async def thinking_react(state: State):
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
    
    last_thought = state["messages"][-1]
    
    metadata_knowledge = {k_id: val["metadata"] for x in state["selected_knowledge"] for k_id, val in x.items()}
    metadata_s_knowledge = {k_id: val["metadata"] for x in state["retrieved_session_knowledge"] for k_id, val in x.items()}
    
    knowledges = [{"chunk": chunk, "metadata": metadata_knowledge[k_id]} for x in state["chunk_knowledge"] for chunk, k_id in [x.values()]]
    s_knowledges = [{"chunk": chunk, "metadata": metadata_s_knowledge[k_id]} for x in state["chunk_retrieved_session_knowledge"] for chunk, k_id in [x.values()]]
    
    memories = [val for x in state["memory"] for _, val in x]
    
    if isinstance(last_thought, ToolMessage):        
        msg = state["last_query"]
        
        system_prompt = prompts.THINKING_REACT_TOOL_SYSTEM_QUERY.format_map({
            "knowledges": knowledges,
            "s_knowledges": s_knowledges,
            "memories": memories,
            "msg": msg,
            "trimmed_msg": trimmed_msg,
        })
        
    else:
        msg = last_thought.content[0]["text"].replace("Thought: QUERY:", "").strip()
        
        system_prompt = prompts.THINKING_REACT_SYSTEM_QUERY.format_map({
            "msg": msg,
            "knowledges": knowledges,
            "s_knowledges": s_knowledges,
            "memories": memories,
            "history": trimmed_msg[:-1],
        })
    
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
            "reasoning_questions_observation": [{"question": msg, "observation": result.content[0]['text']}],
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
    
    system_prompt = prompts.THINKING_END_SYSTEM_QUERY.format_map({
        "knowledges": knowledges,
        "s_knowledges": s_knowledges,
        "memories": memories,
        "reasoning_questions_observation": state["reasoning_questions_observation"]
    })
    
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
    
    prompt = prompts.REASONING_SYSTEM_QUERY.format_map({
        "knowledges": knowledges,
        "s_knowledges": s_knowledges,
        "memories": memories,
        "original_query": trimmed_msg[-1],
        "trimmed_msg_reasoning": trimmed_msg[:-1],
        "iteration": iteration,
    })   
    
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

# if __name__ == "__main__":
    # asyncio.run(get_agent_graph())


# For next development
# Use cron job to update state of the retrieved document ids. delete the id that will have been remove in the State
# Paralellize retrive knowledge/memory
# pisahkan state antara messages ke user dan system messages (ai/tool)
# Menggunakan model khusus coding dan bukan

# react ref: https://machinelearningmastery.com/building-react-agents-with-langgraph-a-beginners-guide/