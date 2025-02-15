from enum import Enum
from typing import List, Type


class ChatStorageType(Enum):
    """Supported chat storage types."""
    DJANGO = "django"
    REDIS = "redis"

class WorkflowType(Enum):
    """Supported workflow types."""
    CHATBOT = "chatbot"


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



class ModelProvider(Enum):
    """Supported model providers."""
    GROQ = "groq"
    OPENAI = "openai"

    def get_model_enum(self) -> Type[BaseModelName]:
        """Return the corresponding model enum class for the provider."""
        if self == ModelProvider.GROQ:
            return GroqModelName
        elif self == ModelProvider.OPENAI:
            return OpenAIModelName
        else:
            raise ValueError(f"Unsupported model provider: {self}")

    @classmethod
    def get_provider_names(cls) -> List[str]:
        """Get list of all available providers."""
        return [provider.value for provider in cls]