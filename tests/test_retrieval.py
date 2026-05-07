import pytest
import os
import importlib
import csv
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric, FaithfulnessMetric
from datasets import load_dataset

load_dotenv()

MODULE_PATH = os.getenv("RETRIEVER_MODULE")
CLASS_NAME = os.getenv("RETRIEVER_CLASS")
BENCHMARK_NAME = os.getenv("BENCHMARK_NAME", "techqa")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
RESULTS_FILE = os.getenv("RESULTS_FILE", "benchmark_performance.tsv")
SAMPLE_SIZE = int(os.getenv("BENCHMARK_SAMPLE_SIZE", 10))

@pytest.fixture(scope="module")
def retriever():
    """Dynamically loads the retriever class defined in .env"""
    try:
        module = importlib.import_module(MODULE_PATH)
        retriever_class = getattr(module, CLASS_NAME)
        return retriever_class(collection_name=f"benchmark_{BENCHMARK_NAME}")
    except (ImportError, AttributeError) as e:
        pytest.exit(f"Critical: Could not load {CLASS_NAME} from {MODULE_PATH}. Error: {e}")

# Load dataset once per session
dataset = load_dataset("galileo-ai/ragbench", BENCHMARK_NAME, split="test").shuffle(seed=42)

@pytest.mark.parametrize("i", range(min(SAMPLE_SIZE, len(dataset))))
def test_retrieval_benchmarks(i, retriever):
    row = dataset[i]
    query = row["question"]
    expected_output = row.get("response") or row.get("answer")

    # Plug-and-play retrieval
    retrieved_contexts, latency = retriever.search(query)

    # Initialize Metrics
    recall_metric = ContextualRecallMetric(threshold=0.7, model=OPENAI_MODEL)
    precision_metric = ContextualPrecisionMetric(threshold=0.7, model=OPENAI_MODEL)
    faithfulness_metric = FaithfulnessMetric(threshold=0.8, model=OPENAI_MODEL)

    test_case = LLMTestCase(
        input=query,
        actual_output=expected_output,
        expected_output=expected_output,
        retrieval_context=retrieved_contexts
    )

    # Execute measures
    try:
        recall_metric.measure(test_case)
        precision_metric.measure(test_case)
        faithfulness_metric.measure(test_case)
    except Exception:
        pass # Logs 0 if the metric fails to execute
    finally:
        save_results(
            i, 
            retriever.label, 
            recall_metric.score, 
            precision_metric.score, 
            faithfulness_metric.score, 
            latency
        )
    
    assert_test(test_case, [recall_metric, precision_metric, faithfulness_metric])

def save_results(idx, config_label, r_score, p_score, f_score, latency):
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        if not file_exists:
            writer.writerow(["Benchmark", "Test_Name", "Configuration", "Recall", "Precision", "Faithfulness", "Latency_ms"])
        
        writer.writerow([
            BENCHMARK_NAME,
            f"row_{idx}",
            config_label,
            round(r_score or 0, 4),
            round(p_score or 0, 4),
            round(f_score or 0, 4),
            round(latency, 2)
        ])