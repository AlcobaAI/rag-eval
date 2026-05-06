import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

# Import your actual agent function here
# from my_agent import run_agent_workflow

def test_v1_rag_pipeline():
    query = "How do I implement a local vector database?"
    
    # 1. Run your agent logic
    # result = run_agent_workflow(query)
    # output = result["answer"]
    # retrieval_context = result["context_chunks"] 
    
    # Mock data for initial Docker test
    output = "You can use Qdrant in a Docker container."
    retrieval_context = ["Qdrant is a vector database that runs well in Docker."]

    # 2. Define metrics
    faith_metric = FaithfulnessMetric(threshold=0.7)
    relevancy_metric = ContextualRelevancyMetric(threshold=0.7)

    # 3. Create the test case
    test_case = LLMTestCase(
        input=query,
        actual_output=output,
        retrieval_context=retrieval_context
    )

    # 4. Evaluate
    assert_test(test_case, [faith_metric, relevancy_metric])