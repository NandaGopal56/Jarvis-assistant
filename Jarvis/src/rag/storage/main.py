from dotenv import load_dotenv
import os
import time
import random
import numpy as np
from pymilvus import DataType

load_dotenv()

class MilvusClientWrapper:
    def __init__(self, uri, token):
        from pymilvus import MilvusClient
        self.client = MilvusClient(uri=uri, token=token)
        print(f"Connected to DB: {uri}")

    def create_collection(self, collection_name, dim):
        schema = self.client.create_schema()
        schema.add_field("book_id", DataType.INT64, is_primary=True, description="customized primary id")
        schema.add_field("word_count", DataType.INT64, description="word count")
        schema.add_field("book_intro", DataType.FLOAT_VECTOR, dim=dim, description="book introduction")
        
        index_params = self.client.prepare_index_params()
        index_params.add_index("book_intro", metric_type="L2")
        
        self.client.create_collection(collection_name, dimension=dim, schema=schema, index_params=index_params)
        print(f"Created collection: {collection_name}")
    
    def insert_data(self, collection_name, nb, insert_rounds):
        start = 0
        total_rt = 0
        for _ in range(insert_rounds):
            vectors = [np.random.rand(64).tolist() for _ in range(nb)]
            rows = [{"book_id": i, "word_count": random.randint(1, 100), "book_intro": vector} for i, vector in enumerate(vectors, start)]
            t0 = time.time()
            self.client.insert(collection_name, rows)
            total_rt += time.time() - t0
            start += nb
        print(f"Inserted {nb * insert_rounds} entities in {round(total_rt, 4)} seconds")
    
    def search_vectors(self, collection_name, dim, nq=1, top_k=1):
        search_params = {"metric_type": "L2", "params": {"level": 2}}
        for i in range(10):
            search_vec = [[random.random() for _ in range(dim)] for _ in range(nq)]
            print(f"Searching vector: {search_vec}")
            t0 = time.time()
            results = self.client.search(collection_name, search_vec, limit=top_k, search_params=search_params, anns_field="book_intro")
            print(f"Result:{results}")
            print(f"search {i} latency: {round(time.time() - t0, 4)} seconds!")

if __name__ == "__main__":
    
    milvus_uri = os.environ.get('MILVUS_ENDPOINT')
    token = os.environ.get('MILVUS_TOKEN')

    milvus = MilvusClientWrapper(milvus_uri, token)
    collection_name = "book"
    dim = 64
    
    if milvus.client.has_collection(collection_name):
        milvus.client.drop_collection(collection_name)
        print(f"Dropped existing collection {collection_name}")
    
    milvus.create_collection(collection_name, dim)
    milvus.insert_data(collection_name, nb=1000, insert_rounds=2)
    milvus.client.flush(collection_name)
    print("Flush completed")
    milvus.search_vectors(collection_name, dim)
