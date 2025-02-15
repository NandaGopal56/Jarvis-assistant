from abc import ABC, abstractmethod
from typing import Any, List, Dict
import chromadb
from chromadb import PersistentClient, Collection
from chromadb.config import Settings
from chromadb.api.types import Metadata

# Abstract Base Class for VectorDB
class VectorDB(ABC):
    """
    Abstract base class for a Vector Database.
    """

    @abstractmethod
    def add_vectors(self, ids: List[str], vectors: List[List[float]], metadatas: List[Metadata]) -> None:
        """
        Add vectors to the database.

        :param ids: List of unique IDs for the vectors.
        :param vectors: List of vector embeddings.
        :param metadatas: List of metadata dictionaries for each vector.
        """
        pass

    @abstractmethod
    def query_vectors(self, query: Dict[str, Any]) -> Any:
        """
        Query vectors in the database.

        :param query: Query object containing search parameters.
        """
        pass

    @abstractmethod
    def delete_vectors(self, ids: List[str]) -> None:
        """
        Delete vectors from the database by IDs.

        :param ids: List of IDs to delete.
        """
        pass

# ChromaDB Implementation of VectorDB
class ChromaDB(VectorDB):
    """
    Implementation of the VectorDB interface using ChromaDB.
    """

    def __init__(self, db_dir: str):
        """
        Initialize ChromaDB with persistent storage.

        :param db_dir: Directory path for persistent storage.
        """
        self.db_dir = db_dir
        self.client = self._create_vector_client()
        self.collection = self._create_collection(self.client, collection_name="vectors")

    def add_vectors(self, ids: List[str], vectors: List[List[float]], metadatas: List[Metadata]) -> None:
        self.collection.add(ids=ids, embeddings=vectors, metadatas=metadatas)

    def query_vectors(self, query: Dict[str, Any]) -> Any:
        return self.collection.query(**query)

    def delete_vectors(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)

    def _create_collection(self, client: PersistentClient, collection_name: str) -> Collection:
        return client.get_or_create_collection(name=collection_name)

    def _create_vector_client(self) -> PersistentClient:
        """
        Private method to create and initialize the ChromaDB client.

        :return: A PersistentClient instance.
        """
        return chromadb.PersistentClient(path=self.db_dir)

# Dependency Injection Example
class VectorDBService:
    """
    Service class that interacts with a VectorDB instance.
    """

    def __init__(self, vector_db: VectorDB):
        """
        Initialize the service with a VectorDB instance.

        :param vector_db: An instance of VectorDB.
        """
        self.vector_db = vector_db

    def add_vectors(self, ids: List[str], vectors: List[List[float]], metadatas: List[Metadata]) -> None:
        """
        Add vectors to the database.

        :param ids: List of IDs.
        :param vectors: List of vector embeddings.
        :param metadatas: List of metadata dictionaries.
        """
        self.vector_db.add_vectors(ids, vectors, metadatas)

    def query_vectors(self, query: Dict[str, Any]) -> Any:
        """
        Query vectors in the database.

        :param query: Query parameters.
        """
        return self.vector_db.query_vectors(query)

    def delete_vectors(self, ids: List[str]) -> None:
        """
        Delete vectors by IDs.

        :param ids: List of IDs to delete.
        """
        self.vector_db.delete_vectors(ids)

# Example Usage
if __name__ == "__main__":
    # Create ChromaDB instance
    chroma_db = ChromaDB("./chromadb_storage")

    # Inject into service
    vector_service = VectorDBService(vector_db=chroma_db)

    # Example data
    ids = ["vec1", "vec2"]
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    metadatas = [{"name": "vector1"}, {"name": "vector2"}]

    # Add vectors
    vector_service.add_vectors(ids, vectors, metadatas)

    # Query vectors
    query = {
        "query_embeddings": [[0.1, 0.2, 0.3]],
        "n_results": 1,
    }
    result = vector_service.query_vectors(query)
    print("Query Result:", result)

    # Delete vectors
    vector_service.delete_vectors(["vec1", "vec2"])
