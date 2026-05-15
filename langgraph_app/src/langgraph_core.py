from models.ollama_qwen import ollama_llm, ollama_embedding

# from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage, HumanMessage
from langchain_core.messages.utils import trim_messages
from langgraph.types import Command
from langgraph.prebuilt import InjectedState, ToolNode
from langchain_core.tools import InjectedToolCallId, tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.documents import Document

from typing_extensions import Annotated, Any
import uuid, copy, numexpr, json
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from utils.documents_utils import encrypt, decrypt, get_vector_store_chroma, get_vector_store_retriever
from src.states import State, RouterOutput, CalculatorExpression
from string_utils.prompts import Prompts

prompts = Prompts()

# Tools
## Tool: Put new memory
@tool
async def put_new_memory( 
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    query: str,
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
    print("Tool: put_new_memory")
    
    tenant_id = state["tenant_id"]
    user_id = state["user_id"]
    memory_id = str(uuid.uuid4())
    
    metadata = {
        "content_type": "memory",
        "len_char": len(query),
        "tenant_id": tenant_id,
        "user_id": user_id,
        "memory_id": memory_id,
        "created_at": str(datetime.now()),
    }
    
    document = Document(
        page_content=query,
        metadata=metadata,
        id=memory_id,
    )
    
    try:
        vector_store = await get_vector_store_chroma(f"user_{user_id.replace('-', '_')}")
        await vector_store.aadd_documents(documents=[document], ids=[memory_id])
        
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
                    "append": [{"memory_id": memory_id}]
                },
                "memory": {
                    "append": [{"memory_id": memory_id, "content": query, "metadata": metadata}]
                },
            }
        )
    
    except Exception as e:
        print(e)
        return "Failed put new memory to the database." 

## Tool: Fetch new knowledge (yang gak ada di state["selected_knowledge"]) 
@tool
async def fetch_new_knowledge(
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    query: str, 
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
    print("Tool: fetch_new_knowledge")
    
    tenant_id = state["tenant_id"]
    selected_knowledge = copy.deepcopy(state["selected_knowledge"])
    
    # Get knowledge ids and chunk ids
    knowledge_ids = [x["knowledge_id"] for x in selected_knowledge]
    chunk_ids = [c_id for x in selected_knowledge for c_id in x["chunk_id"]]
    
    try:
        # Retrieve from the database
        vector_store = await get_vector_store_chroma(f"tenant_{tenant_id.replace('-', '_')}")
        retriever = await get_vector_store_chroma(vector_store)
        results = await retriever.ainvoke(query)
        
        if len(results) == 0:
            return "Success. Based on the query, the knowledge is not exist in the database."
            
        selected_knowledge = {x["knowledge_id"]: {"chunk_ids": x["chunk_ids"]} for x in selected_knowledge}
        knowledge_id_append = []
        chunk_append = []
        replace_ids = set()
        for result in results:
            metadata = result.metadata
            
            chunk_id = metadata["chunk_id"]
            if chunk_id in chunk_ids:
                continue
                
            knowledge_id = metadata["knowledge_id"]
            if knowledge_id not in selected_knowledge_dict.keys():
                selected_knowledge_dict[knowledge_id] = {"chunk_ids": []}
                knowledge_id_append.append(knowledge_id)
            elif knowledge_id in knowledge_ids and knowledge_id not in knowledge_id_append:
                replace_ids.add(knowledge_id)
                
            content = result.page_content
            selected_knowledge_dict[knowledge_id]["chunk_ids"].append(chunk_id)
            chunk_append.append({"chunk_id": chunk_id, "content": content, "metadata": metadata})
            
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="Success fetch new knowledge from the database.", 
                            tool_call_id=tool_call_id,
                            name="fetch_new_knowledge",
                        )
                    ],
                    "selected_knowledge": {
                        "append": [{"knowledge_id": k_id, "chunk_ids": val["chunk_ids"]} for k_id, val in selected_knowledge_dict.items() if k_id in knowledge_id_append],
                        "replace": [{"knowledge_id": k_id, "chunk_ids": val["chunk_ids"]} for k_id, val in selected_knowledge_dict.items() if k_id in replace_ids],
                    },
                    "chunk_knowledge": {
                        "append": chunk_append,
                    },
                }
            )

    except Exception as e:
        print(e)
        return "Failed fetch new knowledge from database."
    
## Tool: Fetch new knowledge_session (yang gak ada di state["retrieved_session_knowledge"])
@tool
async def fetch_new_knowledge_session(
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    query: str, 
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
    print("Tool: fetch_knowledge_session")
    
    thread_id = state["thread_id"]
    retrieved_session_knowledge = copy.deepcopy(state["retrieved_session_knowledge"])
    
    # Get knowledge ids and chunk ids
    s_knowledge_ids = [x["s_knowledge_id"] for x in retrieved_session_knowledge]
    chunk_ids = [c_id for x in retrieved_session_knowledge for c_id in x["chunk_ids"]]
    
    try:
        # Retrieve from the database
        vector_store = await get_vector_store_chroma(f"thread_{thread_id.replace('-', '_')}")
        retriever = await get_vector_store_chroma(vector_store)
        results = await retriever.ainvoke(query)
        
        if len(results) == 0:
            return "Success. Based on the query, the session knowledge is not exist in the database."
            
        retrieved_session_knowledge_dict = {x["s_knowledge_id"]: {"chunk_ids": x["chunk_ids"]} for x in retrieved_session_knowledge}
        s_knowledge_id_append = []
        chunk_append = []
        replace_ids = set()
        for result in results:
            metadata = result.metadata
            
            chunk_id = metadata["chunk_id"]
            if chunk_id in chunk_ids:
                continue
                
            s_knowledge_id = metadata["knowledge_id"]
            if s_knowledge_id not in retrieved_session_knowledge_dict.keys():
                retrieved_session_knowledge_dict[s_knowledge_id] = {"chunk_ids": []}
                s_knowledge_id_append.append(s_knowledge_id)
            elif s_knowledge_id in s_knowledge_ids and s_knowledge_id not in s_knowledge_id_append:
                replace_ids.add(s_knowledge_id)
                
            content = result.page_content
            retrieved_session_knowledge_dict[s_knowledge_id]["chunk_ids"].append(chunk_id)
            chunk_append.append({"chunk_id": chunk_id, "content": content, "metadata": metadata})
            
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
                        "append": [{"s_knowledge_id": k_id, "chunk_ids": val["chunk_ids"]} for k_id, val in retrieved_session_knowledge_dict.items() if k_id in s_knowledge_id_append],
                        "replace": [{"s_knowledge_id": k_id, "chunk_ids": val["chunk_ids"]} for k_id, val in retrieved_session_knowledge_dict.items() if k_id in replace_ids],
                    },
                    "chunk_retrieved_session_knowledge": {
                        "append": chunk_append,
                    },
                }
            )

    except Exception as e:
        print(e)
        return "Failed fetch new knowledge from database."
    
## Tool: Fetch new memory (yang gak ada di state["memory_ids"])
@tool
async def fetch_new_memory( 
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    query: str,
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
    print("Tool: fetch_new_memory")
    
    user_id = state["user_id"]
    memory_ids = copy.deepcopy(state["memory_ids"])
    
    # Get memory ids and chunk ids
    memory_ids_list = [x["memory_id"] for x in memory_ids]
    
    try:
        # Retrieve from the database
        vector_store = await get_vector_store_chroma(f"user_{user_id.replace('-', '_')}")
        retriever = await get_vector_store_chroma(vector_store)
        results = await retriever.ainvoke(query)
        
        if len(results) == 0:
            return "Success. Based on the query, the memory is not exist in the database."
    
        memory_append = []
        memory_ids_append = []
        for result in results:
            metadata = result.metadata
            
            memory_id = metadata["memory_id"]
            if memory_id not in memory_ids_list:
                content = result.page_content
                memory_append.append({"memory_id": memory_id, "content": content})
                memory_id_append.append({"memory_id": memory_id})
                
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
        
    except Exception as e:
        print(e)
        return "Failed fetch new memory from the database."
    
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
    print("Tool: calculator")
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
    print("Tool: web_search")
    try:
        return await search.ainvoke(query)
    except Exception as e:
        print(e)
        return f"Failed to seacrh {query} in the web."

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
    print("Tool: get_datetime_now")
    try:
        return "Year-Month-Date Hour:Minute:Second: " + str(datetime.now(ZoneInfo("Asia/Jakarta")))
    except Exception as e:
        print(e)
        return f"Failed to get current datetime."

## Define Tools node
tools = [put_new_memory, fetch_new_knowledge, fetch_new_memory, fetch_new_knowledge_session, calculator, web_search, get_datetime_now]
tools_by_name = {tool.name: tool for tool in tools}

ollama_llm_tools = gemini_instruct.bind_tools(tools)

tool_node = ToolNode(tools)

async def call_tools(state: State):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = await tools_by_name[tool_call["name"]].ainvoke(tool_call)
        
        if not isinstance(tool_result, Command):
            outputs.append(
                ToolMessage(
                    content=tool_result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
    return {"messages": outputs}
    
async def basic_should_continue(state: State):
    print("Should continue?")
    messages = state["messages"]
    
    tool_calls = getattr(messages[-1], "tool_calls", [])
    
    if len(tool_calls) == 0:
        print("END")
        return END #state["route"] # basic or coding_basic
        
    return "basic_tools"
    
async def coding_basic_should_continue(state: State):
    print("Should continue?")
    messages = state["messages"]

    tool_calls = getattr(messages[-1], "tool_calls", [])

    if len(tool_calls) == 0:
        print("END")
        return END 
        
    return "coding_tools"
    
async def coding_react_should_continue(state: State):
    print("Should continue?")
    messages = state["messages"]

    tool_calls = getattr(messages[-1], "tool_calls", [])

    if len(tool_calls) == 0:
        return "reasoning"
        
    return "coding_react_tools"

async def thinking_react_should_continue(state: State):
    print("Should continue?")
    messages = state["messages"]
    
    tool_calls = getattr(messages[-1], "tool_calls", [])

    if len(tool_calls) == 0:
        return "reasoning"

    return "thinking_react_tools"

# Agents
## Fetch knowledge node if any
async def check_knowledge_exist(state: State):
    """If there is no chunk from database, drop knowledge indices"""
    print("Node: check_knowledge_exist")
    selected_knowledge = copy.deepcopy(state["selected_knowledge"])
    
    if len(selected_knowledge) == 0:
        # No knowledge_id retrieved in this thread_id (conversation)
        return {}
    
    tenant_id = state["tenant_id"]
    knowledge_ids = [x["knowledge_id"] for x in selected_knowledge]
    
    vector_store = await get_vector_store_chroma(f"tenant_{tenant_id.replace('-', '_')}")
    
    # Get knowledge_id data in the database if any
    collection = vector_store._collection
    results = collection.get(
        where={"knowledge_id": {"$in": knowledge_ids}},
        include=["documents", "metadatas"],
    )
   
    if len(results["ids"]) > 0:
        fetched_knowledge_ids = list(set([x["knowledge_id"] for x in results["metadatas"]]))
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
        # Retrieved in state but deleted in the database
        return {
            "selected_knowledge": {
                "remove": item_remove,
        }

    else:
        # Retrieved in state but still in the database (partial or full)
        print("Fetch knowledge...")
        item_append = []
        for index, metadata, chunk in zip(results["ids"], results["metadatas"], results["documents"]):
            if index in selected_knowledge:
                item.append({"chunk_id": index, "content": chunk, "metadata": metadata})
        
        return {
            "selected_knowledge": {
                "remove": item_remove,
            },
            "chunk_knowledge": {
                "append": item_append,
            },
        }
        
async def judge_knowledge(state: State):
    """
    Judge retrieved documents if there are still relevant or not based on history + prompt query
    """
    return {}

## Fetch knowledge_session node if any
async def check_knowledge_session_ttl(state: State):
    """If there is no chunk from database, drop knowledge indices"""
    print("Node: check_knowledge_session_ttl")
    retrieved_session_knowledge = copy.deepcopy(state["retrieved_session_knowledge"])
    
    if len(retrieved_session_knowledge) == 0:
        # No s_knowledge_id retrieved in this thread_id (conversation)
        return {}
    
    thread_id = state["thread_id"]
    s_knowledge_ids = [x["s_knowledge_id"] for x in retrieved_session_knowledge]
    
    vector_store = await get_vector_store_chroma(f"thread_{thread_id.replace('-', '_')}")
    
    # Get s_knowledge_id data in the database if any
    collection = vector_store._collection
    results = collection.get(
        where={"knowledge_id": {"$in": s_knowledge_ids}},
        include=["documents", "metadatas"],
    )
    
    if len(results["ids"]):
        fetched_knowledge_ids = list(set([x["knowledge_id"] for x in results["metadatas"]]))
    else:
        fetch_knowledge_ids = []
        
    item_remove = []
    idx_remove = []
    for i, k_id in enumerate(s_knowledge_ids):
        if k_id not in fetch_knowledge_ids:
            idx_remove.append(i)
            item_remove.append(retrieved_session_knowledge[i])
            
    retrieved_session_knowledge = [x for i, x in enumerate(retrieved_session_knowledge) if i not in idx_remove]
            
    if len(retrieved_session_knowledge) == 0 and len(item_remove) > 0:
        # Retrieved in state but deleted in the database
        return {
            "retrieved_session_knowledge": {
                "remove": item_remove,
            },
        }
        
    else:
        # Retrieved in state but still in the database (partial or full)
        print("Fetch knowledge session...")
        item_append = []
        for index, metadata, chunk in zip(results["ids"], results["metadatas"], results["documents"]):
            if index in retrieved_session_knowledge:
                item.append({"chunk_id": index, "content": chunk, "metadata": metadata})
            
        return {
            "retrieved_session_knowledge": {
                "remove": item_remove,
            },
            "chunk_retrieved_session_knowledge": {
                "append": item_append,
            },
        }
    
async def judge_knowledge_session(state: State):
    """
    Judge retrieved documents if there are still relevant or not based on history + prompt query
    """
    return {}

## Fetch memory node if any
async def check_memory_exist_and_fetch(state: State):
    """If there is no chunk from database, drop memory indices"""
    print("Node: check_memory_exist_and_fetch")
    memory_ids = copy.deepcopy(state["memory_ids"])
    
    if len(memory_ids) == 0:
        # No memory_id retrieved in this thread_id (conversation)
        return {}

    user_id = state["user_id"]
    memory_ids = [x["memory_id"] for x in memory_ids]
    
    vector_store = await get_vector_store_chroma(f"user_{user_id.replace('-', '_')}")
    
    # Get knowledge_id data in the database if any
    collection = vector_store._collection
    results = collection.get(
        where={"memory_id": {"$in": memory_ids}},
        include=["documents", "metadatas"],
    )
    
    if len(results["ids"]) > 0:
        fetched_memory_ids = list(set([x["memory_id"] for x in results["metadatas"]]))
    else:
        fetched_memory_ids = []
    
    item_remove = []
    idx_remove = []
    for i, m_id in enumerate(memory_ids):
        if m_id not in fetched_memory_ids:
            idx_remove.append(i)
            item_remove.append(str(memory_ids[i]))
            
    memory_ids = [x for i, x in enumerate(memory_ids) if i not in idx_remove]
            
    if len(memory_ids) == 0 and len(item_remove) > 0:
        # Retrieved in state but deleted in the database
        return {
            "memory_ids": {
                "remove": item_remove,
            }
        }
    
    else:
        # Retrieved in state but still in the database (partial or full)
        print("Fetch memory...")
        item_append = []
        for index, metadata, chunk in zip(results["ids"], results["metadatas"], results["documents"]):
            if index in memory_ids:
                item_append.append({"memory_id": index, "content": chunk, "metadata": metadata})
            
        return {
            "memory_ids": {
                "remove": item_remove,
            },
            "memory": {
                "append": item_append,
            }
        }

async def judge_memory(state: State):
    """
    Judge retrieved documents if there are still relevant or not based on history + prompt query
    """
    return {}

async def extract_content(message) -> str:
    content = message.content

    if isinstance(content, list): # AI / AI tool call
        content = " ".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        
    if isinstance(message, AIMessage) and message.tool_calls:
        tool_names = [t["name"] for t in message.tool_calls]
        return f"AI: content: {content}, tool calls: {tool_names}"
    
    return content
    
async def trimming_message(messages):
    messages = trim_messages(
        messages,
        strategy="last",
        token_counter=ollama_llm,
        max_tokens=8000,
        start_on="human",
        end_on=("human","tool"),
        include_system=True
    )
    
    trimmed_msg = []
    for msg in messages:
        if msg.type == "tool":
            trimmed_msg.append({"role": msg.type, "content": msg.content})
        else: # human or ai
            extracted_content = await extract_content(msg)
            trimmed_msg.append({"role": msg.type, "content": extracted_content})
            
    return trimmed_msg

## RAG (retrieve data from database based on just new query)
async def rag(state: State):
    print("Node: rag")
    
    tenant_id = state["tenant_id"]
    selected_knowledge = copy.deepcopy(state["selected_knowledge"]) # [{"knowledge_id": knowledge_id, "chunk_ids": [id_1, id_2]}]
   
    # Get knowledge ids and chunk ids
    knowledge_ids = [x["knowledge_id"] for x in selected_knowledge]
    chunk_ids = [c_id for x in selected_knowledge for c_id in x["chunk_id"]]
    
    # Trim messages and convert to dict
    trimmed_msg = await trimming_message(state["messages"])
       
    # Get chunk knowledge
    chunk_knowledge = copy.deepcopy(state["chunk_knowledge"])

    system_query = prompts.RAG_SYSTEM_QUERY.format_map({
        "trimmed_msg_rag": trimmed_msg[:-1],
        "latest_message": trimmed_msg[-1],
        "knowledges": chunk_knowledge,
    })
    
    llm_output = await ollama_llm.ainvoke([SystemMessage(content=system_query)])
    
    # Retrieve from the database
    vector_store = await get_vector_store_chroma(f"tenant_{tenant_id.replace('-', '_')}")
    retriever = await get_vector_store_retriever(vector_store)
    results = await retriever.ainvoke(llm_output.content[0]["text"])
    
    if len(results) == 0:
        return {}

    selected_knowledge_dict = {x["knowledge_id"]: {"chunk_ids": x["chunk_ids"]} for x in selected_knowledge}
    knowledge_id_append = []
    chunk_append = []
    replace_ids = set()
    for result in results:
        metadata = result.metadata
        
        chunk_id = metadata["chunk_id"]
        if chunk_id in chunk_ids:
            continue
            
        knowledge_id = metadata["knowledge_id"]
        if knowledge_id not in selected_knowledge_dict.keys():
            selected_knowledge_dict[knowledge_id] = {"chunk_ids": []}
            knowledge_id_append.append(knowledge_id)    
        elif knowledge_id in knowledge_ids and knowledge_id not in knowledge_id_append:
            replace_ids.add(knowledge_id)
            
        content = result.page_content
        selected_knowledge_dict[knowledge_id]["chunk_ids"].append(chunk_id)
        chunk_append.append({"chunk_id": chunk_id, "content": content, "metadata": metadata})    
            
    return {
        "selected_knowledge": {
            "append": [{"knowledge_id": k_id, "chunk_ids": val["chunk_ids"]} for k_id, val in selected_knowledge_dict.items() if k_id in knowledge_id_append],
            "replace": [{"knowledge_id": k_id, "chunk_ids": val["chunk_ids"]} for k_id, val in selected_knowledge_dict.items() if k_id in replace_ids],
        },
        "chunk_knowledge": {
            "append": chunk_append,
    }

async def judge_rag(state: State):
    """
    Judge retrieved documents if there are still relevant or not based on history + prompt query
    """
    return {}

router_model = ollama_llm.with_structured_output(
    schema=RouterOutput.model_json_schema(), method="json_schema"
)

## Agent: LLM-Instruct/router (fetch knowledge/knowledge_session/memory baru bila perlu)
async def router(state: State): # Tambahkan fungsi atau state untuk format output yang tetap {route: "", mode: ""}
    """Jika prompt user terkait dengan (dokumen) perusahaan, fetch_knowledge. Jika diminta mengingat, fetch user_memory."""
    print("Node: router")
    # Router menentukan mode "thinking" atau "fast" sesuai input user.
    trimmed_msg = await trimming_message(messages)
    
    if state["mode"] == "auto":
        print("auto mode")
        system_query = prompts.AUTO_SYSTEM_QUERY.format_map({
            "trimmed_msg_auto": trimmed_msg[:-1],
            "latest_message": trimmed_msg[-1],
        })
        
    elif state["mode"] == "thinking":
        print("thinking mode")
        system_query = prompts.THINKING_SYSTEM_QUERY.format_map({
            "trimmed_msg_thinking": trimmed_msg[:-1],
            "latest_message": trimmed_msg[-1],
            "format_mode": state["mode"],
        })
        
    else: # fast
        print("fast mode")
        system_query = prompts.FAST_SYSTEM_QUERY.format_map({
            "trimmed_msg_fast": trimmed_msg[:-1],
            "latest_message": trimmed_msg[-1],
            "format_mode": state["mode"],
        })
        
    response = await router_model.ainvoke([SystemMessage(content=system_query)])
    
    if (response["route"] == "coding_react" or response["route"] == "thinking_react"):
        return Command(
            goto="reasoning",
            update={"route": response["route"], "mode": "thinking"}
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
    print("Node: basic")
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=gemini_instruct,
        max_tokens=8000,
        start_on="human",
        end_on=("human","tool"),
        include_system=True
    )
    
    knowledges = [{"content": x["content"], "metadata": x["metadata"]} for x in state["chunk_knowledge"]]
    s_knowledges = [{"content": x["content"], "metadata": x["metadata"]} for x in state["retrieved_session_knowledge"]]
    memories = [{"content": x["content"], "metadata": x["metadata"]} for x in state["memory"]]
    
    system_query = prompts.BASIC_SYSTEM_QUERY.format_map({
        "knowledges": knowledges,
        "s_knowledges": s_knowledges,
        "memories": memories,
    })   
    
    final_query = [SystemMessage(content=system_query), *messages]
    
    if state["streaming_mode"] == True:
        pass
    else:
        response = await ollama_llm_tools.ainvoke(final_query)
    
    print("Berhasil lewat basic")
    return {"messages": [response]}

## Agent: Coding basic
async def coding_basic(state: State):
    print("Node: coding_basic")
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=gemini_instruct,
        max_tokens=8000,
        start_on="human",
        end_on=("human","tool"),
        include_system=True
    )
    
    knowledges = [{"content": x["content"], "metadata": x["metadata"]} for x in state["chunk_knowledge"]]
    s_knowledges = [{"content": x["content"], "metadata": x["metadata"]} for x in state["retrieved_session_knowledge"]]
    memories = [{"content": x["content"], "metadata": x["metadata"]} for x in state["memory"]]
    
    system_query = prompts.CODING_BASIC_SYSTEM_QUERY.format_map({
        "knowledges": knowledges,
        "s_knowledges": s_knowledges,
        "memories": memories,
    })   
    
    final_query = [SystemMessage(content=system_query), *messages]
    
    if state["streaming_mode"] == True:
        pass
    else:
        response = await ollama_llm_tools.ainvoke(final_query)
    
    return {"messages": [response]}
    
## Agent: Coding react
async def coding_react(state: State):
    print("Node: coding_react")
    trimmed_msg = await trimming_message(messages)
    
    last_thought = state["messages"][-1]
    
    knowledges = [{"content": x["content"], "metadata": x["metadata"]} for x in state["chunk_knowledge"]]
    s_knowledges = [{"content": x["content"], "metadata": x["metadata"]} for x in state["retrieved_session_knowledge"]]
    memories = [{"content": x["content"], "metadata": x["metadata"]} for x in state["memory"]]
    
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
        content = last_thought.content
        if isinstance(content, list):
            content = content[0]["text"]
        msg = content.replace("Thought: QUERY:", "").strip()
        
        system_prompt = prompts.CODING_REACT_SYSTEM_QUERY.format_map({
            "msg": msg,
            "knowledges": knowledges,
            "s_knowledges": s_knowledges,
            "memories": memories,
            "history": trimmed_msg[:-1],
        })    
    
    result = await ollama_llm_tools.ainvoke([SystemMessage(content=system_prompt)])
    
    if result.tool_calls:
        return {
            "messages": [result],
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
    print("Node: coding_end")
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=gemini_instruct,
        max_tokens=8000,
        start_on="human",
        end_on=("human","tool"),
        include_system=True
    )
    
    knowledges = [{"content": x["content"], "metadata": x["metadata"]} for x in state["chunk_knowledge"]]
    s_knowledges = [{"content": x["content"], "metadata": x["metadata"]} for x in state["retrieved_session_knowledge"]]
    memories = [{"content": x["content"], "metadata": x["metadata"]} for x in state["memory"]]
    
    system_prompt = prompts.CODING_END_SYSTEM_QUERY.format_map({
        "knowledges": knowledges,
        "s_knowledges": s_knowledges,
        "memories": memories,
        "reasoning_questions_observation": state["reasoning_questions_observation"],
    })
    
    if state["streaming_mode"] == True:
        pass
    else:
        response = await ollama_llm.ainvoke([SystemMessage(content=system_prompt), *messages])
        
    return {"messages": [response]}
    
## Agent: Thinking react
async def thinking_react(state: State):
    print("Node: thinking_react")
    trimmed_msg = await trimming_message(messages)
    
    last_thought = state["messages"][-1]
    
    knowledges = [{"content": x["content"], "metadata": x["metadata"]} for x in state["chunk_knowledge"]]
    s_knowledges = [{"content": x["content"], "metadata": x["metadata"]} for x in state["retrieved_session_knowledge"]]
    memories = [{"content": x["content"], "metadata": x["metadata"]} for x in state["memory"]]
    
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
        content = last_thought.content
        if isinstance(content, list):
            content = content[0]["text"]
        msg = content.replace("Thought: QUERY:", "").strip()
        
        system_prompt = prompts.THINKING_REACT_SYSTEM_QUERY.format_map({
            "msg": msg,
            "knowledges": knowledges,
            "s_knowledges": s_knowledges,
            "memories": memories,
            "history": trimmed_msg[:-1],
        })
    
    result = await ollama_llm_tools.ainvoke([SystemMessage(content=system_prompt)])
    
    if result.tool_calls:
        return {
            "messages": [result],
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
    print("Node: thinking_end")
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=gemini_instruct,
        max_tokens=8000,
        start_on="human",
        end_on=("human","tool"),
        include_system=True
    )
    
    knowledges = [{"content": x["content"], "metadata": x["metadata"]} for x in state["chunk_knowledge"]]
    s_knowledges = [{"content": x["content"], "metadata": x["metadata"]} for x in state["retrieved_session_knowledge"]]
    memories = [{"content": x["content"], "metadata": x["metadata"]} for x in state["memory"]]
    
    system_prompt = prompts.THINKING_END_SYSTEM_QUERY.format_map({
        "knowledges": knowledges,
        "s_knowledges": s_knowledges,
        "memories": memories,
        "reasoning_questions_observation": state["reasoning_questions_observation"]
    })
    
    if state["streaming_mode"] == True:
        pass
    else:
        response = await ollama_llm.ainvoke([SystemMessage(content=system_prompt), *messages])
        
    return {"messages": [response]}
    
## Reasoning node untuk Agent Thinking dan Coding (untuk pengembangan: tambahkan interrupt dan/atau user input sebelum masuk reasoning node)
async def reasoning(state: State):
    print("Node: reasoning")
    
    if state["route"] == "coding_react":
        node_end = "coding_end"
    else: # thinking_react
        node_end = "thinking_end"
    
    iteration = state.get("iteration", 0)
    print(iteration)
    if iteration >= 3:
        return Command(
            goto=node_end,
            update={
                "messages": [AIMessage(content="Thought: I have gathered enough information")],
                "iteration": iteration,
            }
        )
        
    knowledges = [{"content": x["content"], "metadata": x["metadata"]} for x in state["chunk_knowledge"]]
    s_knowledges = [{"content": x["content"], "metadata": x["metadata"]} for x in state["retrieved_session_knowledge"]]
    memories = [{"content": x["content"], "metadata": x["metadata"]} for x in state["memory"]]
            
    trimmed_msg = await trimming_message(messages)
    
    prompt = prompts.REASONING_SYSTEM_QUERY.format_map({
        "knowledges": knowledges,
        "s_knowledges": s_knowledges,
        "memories": memories,
        "original_query": trimmed_msg[-1],
        "trimmed_msg_reasoning": trimmed_msg[:-1],
        "iteration": iteration,
    })   
    
    decision = await ollama_llm.ainvoke([SystemMessage(content=prompt)])
    
    if decision.content[0]["text"].startswith("QUERY:"):
        return Command(
            goto=state["route"],
            update={
                "messages": [AIMessage(content=f"Thought: {decision.content[0]['text']}")],
                "iteration": iteration,
            }
        )
    
    return Command(
        goto=node_end,
        update={
            "messages": [AIMessage(content=f"Thought: {decision.content[0]['text']}")],
            "iteration": iteration,
        }
    )

# Define agent
async def get_agent():
    builder = StateGraph(State)
    
    builder.add_node("check_knowledge_session_ttl", check_knowledge_session_ttl)
    builder.add_node("judge_knowledge_session", judge_knowledge_session)
    builder.add_node("check_knowledge_exist", check_knowledge_exist)
    builder.add_node("judge_knowledge", judge_knowledge)
    builder.add_node("check_memory_exist_and_fetch", check_memory_exist_and_fetch)
    builder.add_node("judge_memory", judge_memory)
    builder.add_node("rag", rag)
    builder.add_node("router", router)
    builder.add_node("basic", basic)
    builder.add_node("coding_basic", coding_basic)
    builder.add_node("coding_react", coding_react)
    builder.add_node("coding_end", coding_end)
    builder.add_node("thinking_react", thinking_react)
    builder.add_node("thinking_end", thinking_end)
    builder.add_node("reasoning", reasoning)
    builder.add_node("basic_tools", tool_node) #call_tools)
    builder.add_node("coding_basic_tools", tool_node)
    builder.add_node("coding_react_tools", tool_node)
    builder.add_node("thinking_react_tools", tool_node)
    
    
    builder.add_edge(START, "check_knowledge_session_ttl")
    builder.add_edge(START, "check_knowledge_exist")
    builder.add_edge(START, "check_memory_exist_and_fetch")
    
    builder.add_edge("check_knowledge_session_ttl", "router")
    builder.add_edge("check_memory_exist_and_fetch", "router")
    builder.add_edge("check_knowledge_exist", "rag")
    builder.add_edge("rag", "router")
    
    # builder.add_edge("check_knowledge_session_ttl", "fetch_knowledge_session")
    # builder.add_edge("check_knowledge_session_ttl", "check_knowledge_exist")
    # builder.add_edge("fetch_knowledge_session", "judge_knowledge_session")
    # builder.add_edge("judge_knowledge_session", "check_knowledge_exist")

    # builder.add_edge("check_knowledge_exist", "fetch_knowledge")
    # builder.add_edge("check_knowledge_exist", "check_memory_exist_and_fetch")
    # builder.add_edge("fetch_knowledge", "judge_knowledge")
    # builder.add_edge("judge_knowledge", "check_memory_exist_and_fetch")
    
    # builder.add_edge("check_memory_exist_and_fetch", "rag")
    # builder.add_edge("check_memory_exist_and_fetch", "judge_memory")
    # builder.add_edge("judge_memory", "rag")
    # builder.add_conditional_edges("check_memory_exist", check_any_documents_uploaded, ["chunk_knowledge_session", "router"])
    # builder.add_conditional_edges("fetch_memory", check_any_documents_uploaded, ["chunk_knowledge_session", "router"])
    # builder.add_edge("check_memory_exist", "fetch_history_messages")
    # builder.add_edge("fetch_memory", "fetch_history_messages")

    #builder.add_conditional_edges("fetch_history_messages", check_any_documents_uploaded, ["chunk_knowledge_session", "router"])
    # builder.add_edge("chunk_knowledge_session", "router")
    # builder.add_edge("rag", "router")
    # builder.add_edge("router", "basic")
    # builder.add_edge("router", "coding_basic")
    # builder.add_edge("router", "reasoning")
    # builder.add_conditional_edges("router", lambda s: s["route"], ["basic", "coding_basic", "reasoning"])
    
    # non-reasoning (basic and coding)
    builder.add_conditional_edges("basic", basic_should_continue, ["basic_tools", END])
    builder.add_edge("basic_tools", "basic")
    
    builder.add_conditional_edges("coding_basic", coding_basic_should_continue, ["coding_basic_tools", END])
    builder.add_edge("coding_basic_tools", "coding_basic")
    
    # thinking-reasoning (coding and thinking)
    # builder.add_edge("reasoning", "coding_react")
    # builder.add_edge("reasoning", "coding_end")
    # builder.add_edge("reasoning", "thinking_react")
    # builder.add_edge("reasoning", "thinking_end")
    # builder.add_conditional_edges("reasoning", lambda s: s["route"], ["coding_react", "coding_end", "thinking_react", "thinking_end"])
    ## coding
    builder.add_conditional_edges("coding_react", coding_react_should_continue, ["coding_react_tools", "reasoning"])
    builder.add_edge("coding_react_tools", "coding_react")
    builder.add_edge("coding_end", END)
    ## thinking
    builder.add_conditional_edges("thinking_react", thinking_react_should_continue, ["thinking_react_tools", "reasoning"])
    builder.add_edge("thinking_react_tools", "thinking_react")
    builder.add_edge("thinking_end", END)
    
    return builder


# For next development
# Use cron job to update state of the retrieved document ids. delete the id that will have been remove in the State
# Paralellize retrive knowledge/memory
# pisahkan state antara messages ke user dan system messages (ai/tool)
# Menggunakan model khusus coding dan bukan

# react ref: https://machinelearningmastery.com/building-react-agents-with-langgraph-a-beginners-guide/