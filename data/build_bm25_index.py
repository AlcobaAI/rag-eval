import argparse
import shutil
from pathlib import Path

import bm25s
import Stemmer
from datasets import load_dataset

BENCHMARKS = {
    "hotpotqa": {
        "path": "hotpot_qa",
        "config": "distractor",
        "split": "validation",
        "field": "context",
    },
    "scifact": {
        "path": "BeIR/scifact",
        "config": "corpus",
        "split": "corpus",
        "field": "text",
    },
    "techqa": {"path": "galileo-ai/ragbench", "config": "techqa", "split": "test", "field": "documents"},
    "finqa": {"path": "galileo-ai/ragbench", "config": "finqa", "split": "test", "field": "documents"},
    "msmarco": {"path": "galileo-ai/ragbench", "config": "msmarco", "split": "test", "field": "documents"},
    "expertqa": {"path": "galileo-ai/ragbench", "config": "expertqa", "split": "test", "field": "documents"},
}


def normalize_document(row, name, field):
    """Converts varying benchmark shapes into a clean list of strings."""
    data = row[field]

    if name == "hotpotqa":
        titles = data["title"]
        sentences_list = data["sentences"]
        paragraphs = []
        for title, sentences in zip(titles, sentences_list):
            full_text = f"Title: {title}\nPassage: " + " ".join(sentences)
            paragraphs.append(full_text)
        return paragraphs

    if isinstance(data, list):
        return [str(item) for item in data if str(item).strip()]

    text = str(data)
    return [text] if text.strip() else []


def build_index_for_benchmark(name, index_prefix, limit, force):
    if name not in BENCHMARKS:
        raise ValueError(f"Unsupported benchmark: {name}")

    cfg = BENCHMARKS[name]
    collection_name = f"benchmark_{name}"
    output_dir = Path(f"{index_prefix}_{collection_name}")

    ds = load_dataset(cfg["path"], cfg["config"], split=cfg["split"])
    rows = ds.select(range(min(limit, len(ds))))

    corpus = []
    for row in rows:
        docs = normalize_document(row, name, cfg["field"])
        corpus.extend(docs)

    if not corpus:
        raise RuntimeError(f"No documents collected for benchmark {name}")

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(corpus, stemmer=stemmer)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(str(output_dir), corpus=corpus)

    print(f"Saved BM25 index for {name} at {output_dir} ({len(corpus)} docs)")


def parse_args():
    parser = argparse.ArgumentParser(description="Build BM25 indices for retrieval benchmarks")
    parser.add_argument(
        "--benchmark",
        default="techqa",
        choices=list(BENCHMARKS.keys()) + ["all"],
        help="Benchmark name or 'all'",
    )
    parser.add_argument(
        "--index-prefix",
        default="indices/bm25",
        help="Index path prefix; final path is '<index-prefix>_benchmark_<name>'",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Max dataset rows to ingest for each benchmark",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing index directory if it exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    targets = list(BENCHMARKS.keys()) if args.benchmark == "all" else [args.benchmark]
    for benchmark_name in targets:
        build_index_for_benchmark(
            name=benchmark_name,
            index_prefix=args.index_prefix,
            limit=args.limit,
            force=args.force,
        )


if __name__ == "__main__":
    main()
