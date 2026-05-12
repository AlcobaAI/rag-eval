import time
import json
import bm25s
import Stemmer
from sentence_transformers import CrossEncoder
from typing import List, Tuple


def _to_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "text" in value and isinstance(value["text"], str):
            return value["text"]
        if "content" in value and isinstance(value["content"], str):
            return value["content"]
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, list):
        return "\n".join(_to_text(item) for item in value)
    return str(value)

class HybridBM25RerankRetriever:
    def __init__(
        self, 
        collection_name: str, 
        index_dir: str = "indices/bm25",
        reranker_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    ):
        self.path = f"{index_dir}_{collection_name}"
        self.label = "Hybrid-BM25-MiniLM-Rerank-K5"
        self.stemmer = Stemmer.Stemmer("english")
        
        # Load lexical index
        try:
            self.retriever = bm25s.BM25.load(self.path, load_corpus=True)
        except Exception as e:
            print(f"Error loading BM25 index at {self.path}: {e}")
            raise
            
        # Load neural reranker
        self.reranker = CrossEncoder(reranker_name)

    def search(self, query: str, limit: int = 5) -> Tuple[List[str], float]:
        start_time = time.perf_counter()
        
        # 1. Lexical Fetch (Over-fetch for reranking)
        # We fetch 20 to give the reranker a good pool to pick from
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
        results, _ = self.retriever.retrieve(query_tokens, k=20)
        
        # results[0] contains the corpus documents (since load_corpus=True)
        passages = results[0].tolist() if results.shape[0] > 0 else []
        passages = [_to_text(p) for p in passages]
        
        if not passages:
            return [], (time.perf_counter() - start_time) * 1000

        # 2. Reranking Step
        # Reranker compares the query against all 20 lexical candidates
        ranks = self.reranker.rank(query, passages)
        
        # 3. Selection
        # Take the top 'limit' (usually 5) based on the reranker's score
        top_indices = [rank['corpus_id'] for rank in ranks[:limit]]
        contexts = [passages[i] for i in top_indices]
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        return contexts, latency_ms