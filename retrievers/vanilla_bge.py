import time
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class QdrantBGERetriever:
    def __init__(self, collection_name, model_name='BAAI/bge-small-en-v1.5', url="http://qdrant:6333"):
        self.client = QdrantClient(url=url)
        self.model = SentenceTransformer(model_name)
        self.collection_name = collection_name
        self.label = f"Vanilla-{model_name.split('/')[-1]}-K5"

    def search(self, query, limit=5):
        start_time = time.perf_counter()
        query_vector = self.model.encode(query).tolist()
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        contexts = [res.payload["text"] for res in results]
        return contexts, latency_ms