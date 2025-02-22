import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Any, Dict
from contextlib import asynccontextmanager
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()


@dataclass
class VectorDBConfig:
    config_dict: Dict[str, Any]

@dataclass
class VectorData:
    id: str
    values: List[float]
    metadata: Dict[str, Any]



class AsyncVectorDBStrategy(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        pass

    @abstractmethod
    async def create_namespace(self, namespace: str, *args, **kwargs) -> bool:
        pass

    @abstractmethod
    async def delete_namespace(self, namespace: str, *args, **kwargs) -> bool:
        pass

    @abstractmethod
    async def upsert_vectors(self, namespace: str, vectors: List[VectorData]) -> bool:
        pass

    @abstractmethod
    async def query_vectors(self, namespace: str, query_vector: List[float], top_k: int = 5) -> List[VectorData]:
        pass

    @abstractmethod
    async def delete_vectors(self, namespace: str, vector_ids: List[str]) -> bool:
        pass


class AsyncPineconeStrategy(AsyncVectorDBStrategy):
    '''Pinecone Strategy Implementation'''
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.pc: Optional[Pinecone] = None

    async def initialize(self) -> None:
        '''Initialize the Pinecone client with the API key'''
        self.pc = Pinecone(api_key=self.config.config_dict['api_key'])

    async def cleanup(self) -> None:
        '''No explicit cleanup is required for Pinecone'''
        pass

    async def create_namespace(self, namespace: str, *args, **kwargs) -> bool:
        '''Pinecone automatically creates namespaces during upsert'''
        return True

    async def delete_namespace(self, namespace: str, *args, **kwargs) -> bool:
        try:
            async with self.pc.IndexAsyncio(host=self.config.config_dict['host']) as idx:
                # Delete all records within the namespace
                await idx.delete(delete_all=True, namespace=namespace)
            return True
        except Exception as e:
            print(f"Error deleting namespace: {e}")
            return False

    async def upsert_vectors(self, namespace: str, vectors: List[VectorData]) -> bool:
        try:
            records = []
            for v in vectors:
                # Create a record merging vector and metadata.
                record = {"id": v.id, "values": v.values, "metadata": v.metadata}
                records.append(record)
            async with self.pc.IndexAsyncio(host=self.config.config_dict['host']) as idx:
                await idx.upsert(namespace=namespace, vectors=records)
            return True
        except Exception as e:
            print(f"Error upserting vectors: {e}")
            return False

    async def query_vectors(self, namespace: str, query_vector: List[float], top_k: int = 5) -> List[VectorData]:
        try:
            async with self.pc.IndexAsyncio(host=self.config.config_dict['host']) as idx:
                # Assume query_records returns a dict with a "matches" key.
                response = await idx.query(namespace=namespace, 
                                           vector=query_vector, 
                                           top_k=top_k, 
                                           include_values=True, 
                                           include_metadata=True
                                        )
                matches = response.get("matches", [])
                results = [
                    VectorData(match["id"], match["values"], match.get("metadata", {}))
                    for match in matches
                ]
            return results
        except Exception as e:
            print(f"Error querying vectors: {e}")
            return []

    async def delete_vectors(self, namespace: str, ids: List[str]) -> bool:
        try:
            async with self.pc.IndexAsyncio(host=self.config.config_dict['host']) as idx:
                await idx.delete(ids=ids, namespace=namespace)
            return True
        except Exception as e:
            print(f"Error deleting vectors: {e}")
            return False


class AsyncVectorDBFactory:
    '''Factory for Creating Database Strategies'''
    @staticmethod
    def create_strategy(config: VectorDBConfig) -> AsyncVectorDBStrategy:
        db_type = config.config_dict.get('db_type')
        if db_type == 'pinecone':
            return AsyncPineconeStrategy(config)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")