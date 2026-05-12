# rag-eval

RAG retrieval evaluation harness for comparing vector-search strategies on public benchmark datasets using Qdrant + BGE embeddings + DeepEval metrics.

## What This Repository Does

This project helps you measure retrieval quality and latency for RAG systems. It provides:

- A benchmark ingestion pipeline that normalizes multiple datasets and indexes them into Qdrant.
- A BM25 indexing pipeline for lexical and hybrid retrieval experiments.
- Pluggable retrievers (vanilla bi-encoder, reranked bi-encoder + cross-encoder, and hybrid BM25 + reranker).
- Evaluation tests powered by DeepEval using contextual recall and contextual precision metrics.
- A Docker-first workflow so you can run benchmarks consistently.

## Repository Structure

- `data/ingest_benchmarks.py`: Downloads benchmark datasets into local cache.
- `data/populate_qdrant.py`: Builds Qdrant collections and uploads embedded passages.
- `data/build_bm25_index.py`: Builds BM25 indices under `indices/` for lexical/hybrid retrieval.
- `data/summarize_test_runs.py`: Summarizes benchmark metrics from `benchmark_performance.tsv`.
- `retrievers/vanilla_bge.py`: Baseline BGE embedding retriever.
- `retrievers/reranked_bge.py`: BGE retriever with cross-encoder reranking.
- `retrievers/hybrid_bm25_rerank.py`: BM25 lexical retrieval with cross-encoder reranking.
- `tests/test_retrieval.py`: Main benchmark evaluation test suite.
- `tests/test_rag.py`: Minimal smoke test scaffold for a full RAG pipeline.
- `benchmark_performance.tsv`: Tab-separated benchmark output (metrics + latency).
- `docker-compose.yml`: Services for evaluator + Qdrant.

## Evaluation Flow

1. Download and cache benchmark datasets.
2. Normalize benchmark documents and index vectors in Qdrant.
3. Build BM25 indices for lexical/hybrid retrievers.
4. Configure which retriever implementation to test via environment variables.
5. Run DeepEval tests against RAGBench samples.
6. Append results to `benchmark_performance.tsv`.

## Benchmarks

The ingestion script includes these benchmark collections:

- `techqa`
- `finqa`
- `msmarco`
- `expertqa`
- `hotpotqa`
- `scifact`

The retrieval test currently loads `galileo-ai/ragbench` with `BENCHMARK_NAME` (default: `techqa`).

## Retrievers

### 1) Vanilla Retriever

- Class: `QdrantBGERetriever`
- File: `retrievers/vanilla_bge.py`
- Behavior: embeds query with `BAAI/bge-small-en-v1.5`, retrieves top `K=5` from Qdrant.

### 2) Reranked Retriever

- Class: `RerankedBGERetriever`
- File: `retrievers/reranked_bge.py`
- Behavior: retrieves top `K=15` candidates with BGE, reranks with `cross-encoder/ms-marco-MiniLM-L-6-v2`, returns top `5`.

### 3) Hybrid BM25 + Reranker Retriever

- Class: `HybridBM25RerankRetriever`
- File: `retrievers/hybrid_bm25_rerank.py`
- Behavior: fetches lexical candidates from BM25 index files, reranks with `cross-encoder/ms-marco-MiniLM-L-6-v2`, returns top `5`.
- Requires prebuilt BM25 index files at paths like `indices/bm25_benchmark_<benchmark>`.

## Requirements

- Python 3.12+
- Docker + Docker Compose
- OpenAI API key (used by DeepEval model-based metrics)

## Setup

### Option A: Docker (recommended)

1. Create a `.env` file in the repo root.
2. Add required variables (example below).
3. Start services:

```bash
docker compose up --build
```

By default, the evaluator container runs:

```bash
uv run deepeval test run tests/test_rag.py
```

You can override the command to run retrieval benchmarks, for example:

```bash
docker compose run --rm evaluator uv run deepeval test run tests/test_retrieval.py
```

For hybrid BM25 runs, build BM25 indices first:

```bash
MSYS_NO_PATHCONV=1 docker compose --env-file .env run --rm -e PYTHONPATH=. -e UV_LINK_MODE=copy evaluator uv run python data/build_bm25_index.py --benchmark all --force
```

On Windows (Git Bash), use:

```bash
MSYS_NO_PATHCONV=1
```

### Option B: Local with uv

```bash
uv sync
uv run python data/ingest_benchmarks.py
uv run python data/populate_qdrant.py
uv run python data/build_bm25_index.py --benchmark all --force
uv run deepeval test run tests/test_retrieval.py
```

When running locally, ensure Qdrant is reachable at the host expected by the retriever (`http://qdrant:6333` by default in code) or adjust retriever initialization accordingly.

## Environment Variables

`tests/test_retrieval.py` relies on these variables:

- `RETRIEVER_MODULE`: Python module path for retriever class.
- `RETRIEVER_CLASS`: Retriever class name to instantiate.
- `BENCHMARK_NAME`: Benchmark config name (default: `techqa`).
- `OPENAI_MODEL`: DeepEval model name (default: `gpt-4o-mini`).
- `BENCHMARK_SAMPLE_SIZE`: Number of shuffled test rows to run (default: `10`).
- `RESULTS_FILE`: Output TSV path (default: `benchmark_performance.tsv`).

Example `.env`:

```env
OPENAI_API_KEY=your_key_here
RETRIEVER_MODULE=retrievers.vanilla_bge
RETRIEVER_CLASS=QdrantBGERetriever
BENCHMARK_NAME=techqa
OPENAI_MODEL=gpt-4o-mini
BENCHMARK_SAMPLE_SIZE=10
RESULTS_FILE=benchmark_performance.tsv
```

To test reranking:

```env
RETRIEVER_MODULE=retrievers.reranked_bge
RETRIEVER_CLASS=RerankedBGERetriever
```

To test hybrid BM25 + reranking:

```env
RETRIEVER_MODULE=retrievers.hybrid_bm25_rerank
RETRIEVER_CLASS=HybridBM25RerankRetriever
```

## Output Format

Results are appended as tab-separated rows with columns:

- `Benchmark`
- `Test_Name`
- `Configuration`
- `Recall`
- `Precision`
- `Latency_ms`

`tests/test_retrieval.py` resumes from previous runs by matching `Benchmark + Test_Name + Configuration` in `benchmark_performance.tsv`. After a run has started, it continues from the next row after the highest completed `row_N` for that benchmark/configuration.

To summarize a results file:

```bash
uv run python data/summarize_test_runs.py
```

You can also point it at a different file:

```bash
uv run python data/summarize_test_runs.py --results-file path/to/other.tsv
```

## Notes

- `tests/test_rag.py` is currently a scaffold with mock output/context and is useful as a smoke test template.
- `tests/test_retrieval.py` is the primary benchmark driver.
- Large model downloads (sentence-transformers + datasets) are cached in Docker volume `hf_cache`.
