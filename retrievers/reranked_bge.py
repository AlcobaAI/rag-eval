import time
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

class RerankedBGERetriever:
    def __init__(self, collection_name, bi_encoder_name='BAAI/bge-small-en-v1.5', reranker_name='cross-encoder/ms-marco-MiniLM-L-6-v2', url="http://qdrant:6333"):
        self.client = QdrantClient(url=url)
        self.bi_encoder = SentenceTransformer(bi_encoder_name)
        self.reranker = CrossEncoder(reranker_name)
        self.collection_name = collection_name
        self.label = f"Reranked-{bi_encoder_name.split('/')[-1]}-MiniLM-K5"

    def search(self, query, limit=15): # Over-fetch for reranking
        start_time = time.perf_counter()
        
        query_vector = self.bi_encoder.encode(query).tolist()
        initial_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        
        passages = [res.payload["text"] for res in initial_results]
        ranks = self.reranker.rank(query, passages)
        
        # Select top 5 after reranking
        top_indices = [rank['corpus_id'] for rank in ranks[:5]]
        contexts = [passages[i] for i in top_indices]
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        return contexts, latency_ms