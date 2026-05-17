from pydantic import BaseModel
from typing_extensions import Literal
import uuid

class AddUserInput(BaseModel):
    user: str
    role: Literal["super_admin", "admin", "user"]
    
class ChatInput(BaseModel):
    tenant_id: str
    user_id: str
    input_prompt: str
    mode: Literal["auto", "thinking", "fast"]
    streaming: bool = False
    thread_id: str | None = None