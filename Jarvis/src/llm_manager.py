from abc import ABC, abstractmethod
from typing import List, Tuple
from enum import Enum
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class GroqModelName(Enum):
    """Supported Groq model names."""
    
    LLAMA_3_2_1B = "llama-3.2-1b-preview"
    LLAMA_3_3_70B = "llama-3.3-70b-versatile"
    MIXTRAL_8X7B = "mixtral-8x7b-32768"
    
    @classmethod
    def get_model_names(cls) -> List[str]:
        """Get list of all available model names."""
        return [model.value for model in cls]

class LanguageModel(ABC):
    """Abstract base class for language model implementations."""
    
    @abstractmethod
    def generate_response(self, messages: List[Tuple[str, str]]) -> str:
        """Generate response from the language model."""
        pass

class GroqLanguageModel(LanguageModel):
    """Groq-specific implementation of the language model."""
    
    def __init__(self, model_name: GroqModelName):
        self._model = ChatGroq(model_name=model_name.value)
    
    def generate_response(self, messages: List[Tuple[str, str]]) -> str:
        return self._model.invoke(input=messages)

class LanguageModelFactory:
    """Factory class to create language model instances."""
    
    @staticmethod
    def create_model(provider: str, model_name: GroqModelName) -> LanguageModel:
        """Create a language model instance for the specified provider.
        
        Args:
            provider: Name of the model provider (e.g., "groq")
            model_name: Enum value from GroqModelName
        Returns:
            Language model instance
        Raises:
            ValueError: If provider is not supported
        """
        if provider.lower() == "groq":
            return GroqLanguageModel(model_name)
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: ['groq']")

# Example usage
if __name__ == "__main__":
    # Create language model instance
    model = LanguageModelFactory.create_model(
        provider="groq",
        model_name=GroqModelName.LLAMA_3_2_1B
    )
    
    messages = [
        ("system", "You are a helpful assistant that knows about indian history. Answer the question under 10 words."),
        ("human", "When did india get its independence?")
    ]
    
    response = model.generate_response(messages)
    print(f"Response: {response.content}\n")