import chromadb
from abc import ABC, abstractmethod
from typing import Any, List, Dict
from chromadb import PersistentClient, Collection
from chromadb.config import Settings
from chromadb.api.types import Metadata

class VectorDatabase(ABC):
    """
    Abstract base class for a Vector Database.
    
    Implements the **Strategy Pattern**, allowing different vector databases to be used interchangeably.
    """

    @abstractmethod
    def add(self, ids: List[str], embeddings: List[List[float]], metadata: List[Metadata]) -> None:
        """
        Add vector embeddings to the database.
        """
        pass

    @abstractmethod
    def query(self, parameters: Dict[str, Any]) -> Any:
        """
        Query vectors in the database.
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """
        Delete vectors from the database.
        """
        pass


class ChromaVectorDatabase(VectorDatabase):
    """
    Concrete implementation of the VectorDatabase interface using ChromaDB.
    
    Uses the **Strategy Pattern** to provide a specific implementation of vector database operations.
    """

    def __init__(self, storage_path: str):
        """
        Initialize ChromaDB with persistent storage.
        """
        self.storage_path = storage_path
        self.client = self._initialize_client()
        self.collection = self._initialize_collection("vectors")

    def add(self, ids: List[str], embeddings: List[List[float]], metadata: List[Metadata]) -> None:
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadata)

    def query(self, parameters: Dict[str, Any]) -> Any:
        return self.collection.query(**parameters)

    def delete(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)

    def _initialize_collection(self, collection_name: str) -> Collection:
        return self.client.get_or_create_collection(name=collection_name)

    def _initialize_client(self) -> PersistentClient:
        """
        Private method to create and initialize the ChromaDB client.
        """
        return chromadb.PersistentClient(path=self.storage_path)


class VectorDatabaseService:
    """
    Facade class that provides a simplified interface to interact with a Vector Database.
    
    Implements the **Facade Pattern**, hiding complex database operations behind a unified interface.
    """

    def __init__(self, database: VectorDatabase):
        """
        Initialize the facade with a VectorDatabase instance.
        """
        self.database = database

    def add_vectors(self, ids: List[str], embeddings: List[List[float]], metadata: List[Metadata]) -> None:
        """
        Adds vector embeddings to the database.
        """
        self.database.add(ids, embeddings, metadata)

    def query_vectors(self, parameters: Dict[str, Any]) -> Any:
        """
        Queries vector embeddings from the database.
        """
        return self.database.query(parameters)

    def delete_vectors(self, ids: List[str]) -> None:
        """
        Deletes vector embeddings from the database by IDs.
        """
        self.database.delete(ids)


# Example Usage
if __name__ == "__main__":
    # Create ChromaDB instance
    chroma_db = ChromaVectorDatabase("./chromadb_storage")

    # Inject into the facade
    vector_service = VectorDatabaseService(database=chroma_db)

    # Example data
    ids = ["vec1", "vec2"]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    metadata = [{"name": "vector1"}, {"name": "vector2"}]

    # Add vectors
    vector_service.add_vectors(ids, embeddings, metadata)

    # Query vectors
    query_parameters = {
        "query_embeddings": [[0.1, 0.2, 0.3]],
        "n_results": 1,
    }
    result = vector_service.query_vectors(query_parameters)
    print("Query Result:", result)

    # Delete vectors
    vector_service.delete_vectors(["vec1", "vec2"])
