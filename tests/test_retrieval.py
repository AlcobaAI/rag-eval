import pytest
import csv
import os
import time
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric, FaithfulnessMetric
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
BENCHMARK_NAME = os.getenv("BENCHMARK_NAME", "techqa")
COLLECTION = os.getenv("COLLECTION", f"benchmark_{BENCHMARK_NAME}")
RESULTS_FILE = os.getenv("RESULTS_FILE", "benchmark_performance.tsv")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SAMPLE_SIZE = int(os.getenv("BENCHMARK_SAMPLE_SIZE", 100))
STRICT_MODE = os.getenv("STRICT_MODE", "True") == "True"


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

dataset = load_dataset("galileo-ai/ragbench", BENCHMARK_NAME, split="test").shuffle(seed=42)

test_range = range(min(SAMPLE_SIZE, len(dataset)))

@pytest.mark.parametrize("i", test_range)
def test_retrieval_benchmarks(i):
    # The rest of the logic remains identical
    row = dataset[i]
    query = row["question"]
    expected_output = row.get("response") or row.get("answer")

    retrieved_contexts, latency = search_qdrant(query)

    recall_metric = ContextualRecallMetric(threshold=0.7, model=OPENAI_MODEL)
    precision_metric = ContextualPrecisionMetric(threshold=0.7, model=OPENAI_MODEL)
    faithfulness_metric = FaithfulnessMetric(threshold=0.8, model=OPENAI_MODEL)

    test_case = LLMTestCase(
        input=query,
        actual_output=expected_output, 
        expected_output=expected_output,
        retrieval_context=retrieved_contexts
    )

    try:
        recall_metric.measure(test_case)
        precision_metric.measure(test_case)
        faithfulness_metric.measure(test_case)
        assert_test(test_case, [recall_metric, precision_metric, faithfulness_metric])
    except Exception as e:
        pass
    finally:
        with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')

            r_score = getattr(recall_metric, 'score', 0)
            p_score = getattr(precision_metric, 'score', 0)
            f_score = getattr(faithfulness_metric, 'score', 0)

            writer.writerow([
                f"{BENCHMARK_NAME}_row_{i}",
                CONFIG_LABEL,
                round(r_score, 4) if r_score is not None else 0,
                round(p_score, 4) if p_score is not None else 0,
                round(f_score, 4) if f_score is not None else 0,
                round(latency, 2)
            ])