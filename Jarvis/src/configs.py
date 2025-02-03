from enum import Enum
from typing import List


class ChatStorageType(Enum):
    """Supported chat storage types."""
    DJANGO = "django"
    REDIS = "redis"

class WorkflowType(Enum):
    """Supported workflow types."""
    CHATBOT = "chatbot"

class ModelProvider(Enum):
    """Supported model providers."""
    GROQ = "groq"

    @classmethod
    def get_provider_names(cls) -> List[str]:
        """Get list of all available model names."""
        return [provider.value for provider in cls]

class BaseModelName(str, Enum):
    """Base class for model names."""
    
    @classmethod
    def get_model_names(cls) -> List[str]:
        """Get list of all available model names."""
        return [model.value for model in cls]

class GroqModelName(BaseModelName):
    """Supported Groq model names."""
    
    LLAMA_3_2_1B = "llama-3.2-1b-preview"
    LLAMA_3_3_70B = "llama-3.3-70b-versatile"
    MIXTRAL_8X7B = "mixtral-8x7b-32768"

class OpenAIModelName(BaseModelName):
    """Supported OpenAI model names."""
    
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo-preview"