from abc import ABC, abstractmethod
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Abstract Base Class for LLM Invocation
class LLMInvoker(ABC):
    """
    Abstract base class for invoking LLM.
    """

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """
        Generate text based on the given prompt.

        :param prompt: Input prompt for the LLM.
        :return: Generated text.
        """
        pass


# Concrete Implementation of LLMInvoker using Groq
class GroqLLMInvoker(LLMInvoker):
    """
    LLM invoker using Groq for LLM invocation.
    """

    def __init__(self, model_name: str):
        """
        Initialize the Groq LLM invoker.

        :param model_name: The name of the model to use for the LLM.
        """
        self.llm_model = ChatGroq(model_name=model_name)

    def invoke(self, prompt: str) -> str:
        """
        Generate text based on the given prompt using Groq.

        :param prompt: Input prompt for the LLM.
        :return: Generated text.
        """
        return self.llm_model.invoke(input=prompt)


# Factory Class for LLM and Embedding Client
class LLMResponseClientFactory:
    """
    Factory class to create LLM and embeddings clients based on provider selection.
    """

    @staticmethod
    def create_llm_invoker(provider: str, model_name: str) -> LLMInvoker:
        """
        Create an LLM invoker based on the provider (e.g., Groq).

        :param provider: The provider for the LLM (e.g., "groq").
        :param model_name: The name of the model for LLM invocation.
        :return: An instance of a class implementing the LLMInvoker interface.
        """
        if provider == "groq":
            return GroqLLMInvoker(model_name)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


# Example usage
if __name__ == "__main__":
    """
    Example usage of the LLM invoker and embeddings generator.
    """

    # Create LLM invoker with Groq
    llm_invoker = LLMResponseClientFactory.create_llm_invoker(provider="groq", model_name="llama-3.2-1b-preview")

    # Invoke LLM with a prompt
    messages = [
        (
            "system",
            "You are a helpful assistant that knows about indian history. Answer the question under 10 words.",
        ),
        (
            "human", 
            "When did india get its independence?"
        ),
    ]
    response = llm_invoker.invoke(messages)
    print(f"LLM Response: {response.content}\n")