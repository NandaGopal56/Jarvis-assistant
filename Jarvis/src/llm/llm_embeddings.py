from abc import ABC, abstractmethod
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Abstract Base Class for Embeddings Generation
class EmbeddingsGenerator(ABC):
    """
    Abstract base class for generating embeddings.
    """

    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of input texts.

        :param texts: A list of input strings to generate embeddings for.
        :return: A list of embeddings, where each embedding is a list of floats.
        """
        pass


# Concrete Implementation of EmbeddingsGenerator using Gemini
class GeminiEmbeddingsGenerator(EmbeddingsGenerator):
    """
    Embeddings generator using Gemini for embeddings generation.
    """

    def __init__(self, model_name: str):
        """
        Initialize the Gemini embeddings generator.

        :param model_name: The name of the embedding model.
        """
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=model_name)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of input texts using Gemini.

        :param texts: A list of input strings to generate embeddings for.
        :return: A list of embeddings, where each embedding is a list of floats.
        """
        return self.embedding_model.embed_documents(texts)


# Factory Class for LLM and Embedding Client
class LLMEmbeddingsClientFactory:
    """
    Factory class to create LLM and embeddings clients based on provider selection.
    """

    @staticmethod
    def create_embeddings_generator(provider: str, model_name: str) -> EmbeddingsGenerator:
        """
        Create an embeddings generator based on the provider (e.g., Gemini).

        :param provider: The provider for the embeddings generation (e.g., "gemini").
        :param model_name: The name of the model for embeddings generation.
        :return: An instance of a class implementing the EmbeddingsGenerator interface.
        """
        if provider == "gemini":
            return GeminiEmbeddingsGenerator(model_name)
        else:
            raise ValueError(f"Unsupported embeddings provider: {provider}")



# Example usage
if __name__ == "__main__":
    """
    Example usage of the LLM invoker and embeddings generator.
    """

    # Create Embeddings generator with Gemini
    embeddings_generator = LLMEmbeddingsClientFactory.create_embeddings_generator(provider="gemini", model_name="models/embedding-001")

    # Generate embeddings
    texts = ["Hello, how are you?", "This is a test of vector embeddings."]
    embeddings = embeddings_generator.generate_embeddings(texts)

    # Output embeddings
    for i, embedding in enumerate(embeddings):
        print(f"Text: {texts[i]}\nEmbedding: {embedding[:5]}...\n")  # Printing only the first 5 dimensions for brevity

