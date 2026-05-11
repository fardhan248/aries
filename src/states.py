from langchain_core.messages import BaseMessage

import copy, operator, uuid
from typing_extensions import TypedDict, Annotated, Literal, Any
from pydantic import BaseModel, Field

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
    tenant_id: str #uuid.UUID
    user_id: str #uuid.UUID
    thread_id: str #uuid.UUID # thread_id
    mode: Literal["auto", "thinking", "fast"] = "auto" # auto, thinking (reasoning), fast (no reasoning)
    streaming_mode: bool = False

    messages: Annotated[list[BaseMessage], operator.add] = [] # list of AnyMessage, Human, AI, Tool, System
    selected_knowledge: Annotated[list[dict], items_reducer] = [] # list of dict: [{knowledge_id: {"metadata": metadata, "chunk_ids": [id_1, id_2]}}]
    chunk_knowledge: Annotated[list[dict], items_reducer] = [] # list of dict: [{chunk_id: content, "knowledge_id": knowledge_id}]
    retrieved_session_knowledge: Annotated[list[dict], items_reducer] = [] # list of dict: [{s_knowledge_id: {"metadata": metadata, "chunk_ids": [id_1, id_2]}}]
    chunk_retrieved_session_knowledge: Annotated[list[dict], items_reducer] = [] # list of dict: [{chunk_id: content, "s_knowledge_id": s_knowledge_id}]
    memory_ids: Annotated[list, items_reducer] = [] # list of str: [memory_id]
    memory: Annotated[list[dict], items_reducer] = [] # list of dict: [{memory_id: content}]
    
    last_query: str
    iteration: int = 0 # For reasoning iteration
    route: Literal["basic", "coding_basic", "coding_react", "thinking_react"] # basic, coding-basic, coding-reasoning, thinking-reasoning. 
                                                                              # router akan memilih antara [basic, coding_basic, coding_react, thinking_react]
    reasoning_questions_observation: Annotated[list[dict[str, str]], operator.add] = []       

class RouterOutput(BaseModel):
    route: Literal["basic", "coding_basic", "coding_react", "thinking_react"]

class CalculatorExpression(BaseModel):
    expr: str
    variables: dict[str, Any] = Field(default_factory=dict)