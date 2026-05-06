import os
from datasets import load_dataset

def download():
    # The "Core Four" for a balanced Agentic RAG evaluation
    configs = ["techqa", "finqa", "msmarco", "expertqa"]
    
    print("Starting Galileo RAGBench Bulk Download...")
    for config in configs:
        print(f"--- Downloading {config} ---")
        try:
            load_dataset("galileo-ai/ragbench", config)
            print(f"Successfully cached {config}")
        except Exception as e:
            print(f"Failed to download {config}: {e}")

    print("Checking HotpotQA and SciFact...")
    load_dataset("hotpot_qa", "distractor")
    load_dataset("BeIR/scifact", "corpus")

if __name__ == "__main__":
    download()