from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

import copy, operator, uuid
from typing_extensions import TypedDict, Annotated, Literal, Any
from pydantic import BaseModel, Field

# State
def items_reducer(current: list, new: dict | list):
    if current is None:
        current = []
        
    result = copy.deepcopy(current)
    
    # Fan-in 
    if isinstance(new, list):
        if any(isinstance(i, dict) and any(k in i for k in ("append", "replace", "remove")) for i in new):
            for update in new:
                result = items_reducer(result, update)
            return result
        else:
            new = [{"append": new}]
    
    # Remove element
    for item in new.get("remove", []):
        if item in result:
            result.remove(item)
    
    # Append element
    for item in new.get("append", []):
        if item not in result:
            result.append(item)
            
    # Replace element (especially for selected knowledge_id/s_knowledge_id)
    for item in new.get("replace", []):
        if isinstance(item, dict):
            _id = list(item.keys())[0] # knowledge_id
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
    thread_id: str 
    mode: Literal["auto", "thinking", "fast"] = "auto" # auto, thinking (reasoning), fast (no reasoning)
    streaming_mode: bool = False

    messages: Annotated[list[BaseMessage], add_messages] = [] # list of AnyMessage, Human, AI, Tool, System
    selected_knowledge: Annotated[list[dict[str, Any]], items_reducer] = [] # list of dict: [{"knowledge_id": knowledge_id, "chunk_ids": [id_1, id_2]}]
    chunk_knowledge: Annotated[list[dict[str, Any]], items_reducer] = [] # list of dict: [{"chunk_id": chunk_id, "content": content, "metadata": metadata}]
    retrieved_session_knowledge: Annotated[list[dict[str, Any]], items_reducer] = [] # list of dict: [{"s_knowledge_id": s_knowledge_id, "chunk_ids": [id_1, id_2]}]
    chunk_retrieved_session_knowledge: Annotated[list[dict[str, Any]], items_reducer] = [] # list of dict: [{"chunk_id": chunk_id, "content": content, "metadata": metadata}]
    memory_ids: Annotated[list[dict[str, str]], items_reducer] = [] # list of str: [{"memory_id": memory_id}]
    memory: Annotated[list[dict[str, Any]], items_reducer] = [] # list of dict: [{"memory_id": memory_id, "content": content, "metadata": metadata}]
    
    last_query: str = ""
    iteration: int = 0 # For reasoning iteration
    route: Literal["basic", "coding_basic", "coding_react", "thinking_react"] # basic, coding-basic, coding-reasoning, thinking-reasoning. 
                                                                              # router akan memilih antara [basic, coding_basic, coding_react, thinking_react]
    reasoning_questions_observation: Annotated[list[dict[str, str]], operator.add] = []       

class RouterOutput(BaseModel):
    route: Literal["basic", "coding_basic", "coding_react", "thinking_react"]

class CalculatorExpression(BaseModel):
    expr: str
    variables: dict[str, Any] = Field(default_factory=dict)