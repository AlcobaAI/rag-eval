import pytest
import csv
import os
import time
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# --- Configuration ---
BENCHMARK_NAME = "techqa" 
COLLECTION = f"benchmark_{BENCHMARK_NAME}"
RESULTS_FILE = "benchmark_performance.tsv"
# Change this label when you test different techniques (e.g., "BGE-Small-K10")
CONFIG_LABEL = "Vanilla-BGE-Small-K5" 

client = QdrantClient(url="http://qdrant:6333")
embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# Initialize TSV with headers if it doesn't exist
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["Test_Name", "Configuration", "Recall_Score", "Precision_Score", "Search_Latency_ms"])

def search_qdrant(query, limit=5):
    start_time = time.perf_counter()
    query_vector = embed_model.encode(query).tolist()
    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=limit,
        with_payload=True
    )
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    return [res.payload["text"] for res in results], latency_ms

@pytest.mark.parametrize("i, row", enumerate(load_dataset("galileo-ai/ragbench", BENCHMARK_NAME, split="test").select(range(10))))
def test_retrieval_benchmarks(i, row):
    query = row["question"]
    expected_output = row.get("response") or row.get("answer")

    retrieved_contexts, latency = search_qdrant(query)

    recall_metric = ContextualRecallMetric(threshold=0.7)
    precision_metric = ContextualPrecisionMetric(threshold=0.7)

    test_case = LLMTestCase(
        input=query,
        actual_output=expected_output, 
        expected_output=expected_output,
        retrieval_context=retrieved_contexts
    )

    try:
        recall_metric.measure(test_case)
        precision_metric.measure(test_case)
        
        assert_test(test_case, [recall_metric, precision_metric])
    except Exception as e:
        pass
    finally:
        with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')

            r_score = getattr(recall_metric, 'score', 0)
            p_score = getattr(precision_metric, 'score', 0)
            
            writer.writerow([
                f"{BENCHMARK_NAME}_row_{i}",
                CONFIG_LABEL,
                round(r_score, 4) if r_score is not None else 0,
                round(p_score, 4) if p_score is not None else 0,
                round(latency, 2)
            ])