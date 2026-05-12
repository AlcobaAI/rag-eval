import pytest
import os
import importlib
import csv
import json
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
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

dataset = load_dataset("galileo-ai/ragbench", BENCHMARK_NAME, split="test").shuffle(seed=42)


def to_text(value):
    """Normalize arbitrary dataset/retriever values into plain text for DeepEval."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        if "text" in value and isinstance(value["text"], str):
            return value["text"]
        if "content" in value and isinstance(value["content"], str):
            return value["content"]
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, list):
        return "\n".join(to_text(item) for item in value)
    return str(value)

def get_test_indices(benchmark_name, config_label):
    """Get indices of rows that haven't been processed yet for this benchmark/config combo."""
    processed_indices = set()
    max_processed_index = -1
    
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)  # Skip header
            for row in reader:
                if row and len(row) >= 3:
                    benchmark = row[0]
                    test_name = row[1]
                    configuration = row[2]
                    # Only mark as processed if benchmark, test_name, AND configuration all match
                    if benchmark == benchmark_name and configuration == config_label and test_name.startswith("row_"):
                        try:
                            idx = int(test_name.split("_")[1])
                            processed_indices.add(idx)
                            if idx > max_processed_index:
                                max_processed_index = idx
                        except (ValueError, IndexError):
                            pass
    
    if max_processed_index >= 0:
        candidate_indices = list(range(max_processed_index + 1, len(dataset)))
    else:
        candidate_indices = list(range(min(SAMPLE_SIZE, len(dataset))))
    
    unprocessed = [i for i in candidate_indices if i not in processed_indices]
    
    return unprocessed

def pytest_generate_tests(metafunc):
    """Dynamically generate test parameters and skip if all samples are processed"""
    if "i" in metafunc.fixturenames:
        # Load retriever to get configuration label
        try:
            module = importlib.import_module(MODULE_PATH)
            retriever_class = getattr(module, CLASS_NAME)
            retriever_instance = retriever_class(collection_name=f"benchmark_{BENCHMARK_NAME}")
            config_label = retriever_instance.label
        except (ImportError, AttributeError) as e:
            pytest.exit(f"Critical: Could not load {CLASS_NAME} from {MODULE_PATH}. Error: {e}")
        
        indices = get_test_indices(BENCHMARK_NAME, config_label)
        if not indices:
            pytest.skip(f"All {SAMPLE_SIZE} samples have been processed for {BENCHMARK_NAME} with configuration {config_label}. No new tests to run.")
        metafunc.parametrize("i", indices)

def test_retrieval_benchmarks(i, retriever):
    row = dataset[i]
    query = to_text(row["question"])
    expected_output = to_text(row.get("response") or row.get("answer"))

    retrieved_contexts, latency = retriever.search(query)
    retrieved_contexts = [to_text(ctx) for ctx in (retrieved_contexts or [])]

    recall_metric = ContextualRecallMetric(threshold=0.7, model=OPENAI_MODEL)
    precision_metric = ContextualPrecisionMetric(threshold=0.7, model=OPENAI_MODEL)

    test_case = LLMTestCase(
        input=query,
        actual_output=expected_output,
        expected_output=expected_output,
        retrieval_context=retrieved_contexts
    )

    try:
        recall_metric.measure(test_case)
        precision_metric.measure(test_case)
    except Exception:
        pass
    finally:
        save_results(
            i, 
            retriever.label, 
            recall_metric.score, 
            precision_metric.score, 
            latency
        )
    
    assert_test(test_case, [recall_metric, precision_metric])

def save_results(idx, config_label, r_score, p_score, latency):
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        if not file_exists:
            writer.writerow(["Benchmark", "Test_Name", "Configuration", "Recall", "Precision", "Latency_ms"])
        
        writer.writerow([
            BENCHMARK_NAME,
            f"row_{idx}",
            config_label,
            round(r_score or 0, 4),
            round(p_score or 0, 4),
            round(latency, 2)
        ])
