
import unittest
import os
import warnings
import constants
from dotenv import load_dotenv
from vector_store import VectorDBConfig, AsyncVectorDBFactory, VectorData


load_dotenv()



class AsyncTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Set up a test configuration. Replace with real values if needed.
        self.config = VectorDBConfig(config_dict={
            'api_key': os.environ.get('PINECONE_API_KEY'),
            'host': os.environ.get('PINECONE_HOST'),
            'db_type': 'pinecone'
        })
        self.strategy = AsyncVectorDBFactory.create_strategy(self.config)
        await self.strategy.initialize()

    async def asyncTearDown(self):
        await self.strategy.cleanup()

    async def test_create_namespace(self):
        result = await self.strategy.create_namespace("test_namespace")
        self.assertTrue(result)

    async def test_upsert_vectors(self):
        vectors = [
            VectorData(id="1", values=constants.vector_1, metadata={"field": "value"}),
            VectorData(id="2", values=constants.vector_2, metadata={"field": "value2"})
        ]
        result = await self.strategy.upsert_vectors("test_namespace", vectors)
        self.assertTrue(result)

    async def test_query_vectors(self):
        results = await self.strategy.query_vectors("test_namespace", constants.vector_2, top_k=2)
        # We only check that results is a list; detailed checking depends on the live data.
        self.assertIsInstance(results, list)

    async def test_delete_vectors(self):
        # upsert data to create namespace
        vectors = [
            VectorData(id="1", values=constants.vector_1, metadata={"field": "value"}),
            VectorData(id="2", values=constants.vector_2, metadata={"field": "value2"})
        ]
        result = await self.strategy.upsert_vectors("test_namespace", vectors)

        # delete vectors
        result = await self.strategy.delete_vectors(namespace="test_namespace", ids=["1", "2"])
        self.assertTrue(result)

    async def test_delete_namespace(self):
        # upsert data to create namespace
        vectors = [
            VectorData(id="1", values=constants.vector_1, metadata={"field": "value"}),
            VectorData(id="2", values=constants.vector_2, metadata={"field": "value2"})
        ]
        result = await self.strategy.upsert_vectors("test_namespace", vectors)

        # delete namespace
        result = await self.strategy.delete_namespace("test_namespace")
        self.assertTrue(result)

if __name__ == '__main__':
    warnings.filterwarnings(action="ignore", message="Enable", category=ResourceWarning)

    unittest.main()
