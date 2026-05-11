from pydantic import BaseModel
from typing_extensions import Literal
import uuid

class AddUserInput(BaseModel):
    user: str
    role: Literal["super_admin", "admin", "user"]
    
class ChatInput(BaseModel):
    tenant_id: uuid.UUID
    user_id: uuid.UUID
    input_prompt: str
    mode: Literal["auto", "thinking", "fast"]
    streaming: bool = False
    thread_id: uuid.UUID | None = None
    
# class NewChat(BaseModel):
    # tenant_id: uuid.UUID
    # user_id: uuid.UUID
    # input_prompt: str
    # mode: Literal["auto", "thinking", "fast"]
    # streaming: bool = False
    # thread_id: uuid.UUID | None = None