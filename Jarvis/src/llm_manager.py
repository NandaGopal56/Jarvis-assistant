from abc import ABC, abstractmethod
from typing import List, Tuple
from enum import Enum
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from src.configs import BaseModelName, GroqModelName, OpenAIModelName, ModelProvider

# Load environment variables from .env file
load_dotenv()


class LanguageModel(ABC):
    """Abstract base class for language model implementations."""
    
    @abstractmethod
    def _validate_model_name(self, model_name) -> None:
        """Validate that the provided model name is supported.
        
        Args:
            model_name: The model name to validate
            
        Raises:
            ValueError: If the model name is not supported
        """
        pass
    
    @abstractmethod
    def generate_response(self, messages: List[Tuple[str, str]]) -> str:
        """Generate response from the language model."""
        pass

class GroqLanguageModel(LanguageModel):
    """Groq-specific implementation of the language model."""
    
    def __init__(self, model_name: GroqModelName):
        self._validate_model_name(model_name)
        self._model = ChatGroq(model_name=model_name.value)
    
    def _validate_model_name(self, model_name: GroqModelName) -> None:
        """Validate that the provided model name is supported by Groq"""
        
        supported_models = set(GroqModelName.get_model_names())
        if model_name.value not in supported_models:
            raise ValueError(
                f"Unsupported Groq model: {model_name.value}. "
                f"Supported models: {', '.join(supported_models)}"
            )
    
    def generate_response(self, messages: List[Tuple[str, str]]) -> str:
        return self._model.invoke(input=messages)



class LanguageModelFactory:
    """Factory class to create language model instances."""
    
    @staticmethod
    def create_model(provider: str, model_name: BaseModelName) -> LanguageModel:
        """Create a language model instance for the specified provider.
        
        Args:
            provider: Name of the model provider (e.g., "groq")
            model_name: Enum value from GroqModelName
        Returns:
            Language model instance
        Raises:
            ValueError: If provider is not supported
        """
        if provider == ModelProvider.GROQ:
            return GroqLanguageModel(model_name)
        
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: [{ModelProvider.get_provider_names()}]")

# Example usage
if __name__ == "__main__":
    # Create language model instance
    model = LanguageModelFactory.create_model(
        provider=ModelProvider.GROQ,
        model_name=GroqModelName.LLAMA_3_2_1B
    )
    
    messages = [
        ("system", "You are a helpful assistant that knows about indian history. Answer the question under 10 words."),
        ("human", "When did india get its independence?")
    ]
    
    response = model.generate_response(messages)
    print(f"Response: {response.content}\n")