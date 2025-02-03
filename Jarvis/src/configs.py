from enum import Enum
from dataclasses import dataclass
from src.llm_manager import GroqModelName


class ChatStorageType(Enum):
    DJANGO = "django"
    REDIS = "redis"

class WorkflowType(Enum):
    CHAT = "chat"

@dataclass
class ChatConfig:
    """Configuration for chat settings"""
    model_provider: str = "groq"
    model_name: GroqModelName = GroqModelName.LLAMA_3_2_1B
    storage_type: ChatStorageType = ChatStorageType.DJANGO
    temperature: float = 0