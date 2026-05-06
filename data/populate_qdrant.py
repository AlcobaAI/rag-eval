import os
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-small-en-v1.5')
VECTOR_SIZE = 384 

client = QdrantClient(host="qdrant", port=6333)

BENCHMARKS = {
    "hotpotqa": {
        "path": "hotpot_qa", 
        "config": "distractor", 
        "split": "validation", 
        "field": "context"
    },
    "scifact": {
        "path": "BeIR/scifact", 
        "config": "corpus", 
        "split": "corpus", 
        "field": "text"
    },
    "techqa": {"path": "galileo-ai/ragbench", "config": "techqa", "split": "test", "field": "documents"},
    "finqa": {"path": "galileo-ai/ragbench", "config": "finqa", "split": "test", "field": "documents"},
    "msmarco": {"path": "galileo-ai/ragbench", "config": "msmarco", "split": "test", "field": "documents"},
    "expertqa": {"path": "galileo-ai/ragbench", "config": "expertqa", "split": "test", "field": "documents"}
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
        return [str(item) for item in data]

    return [str(data)]

def ingest_all():
    for name, cfg in BENCHMARKS.items():
        collection_name = f"benchmark_{name}"
        print(f"📦 Normalizing and Ingesting: {name}")

        try:
            ds = load_dataset(cfg["path"], cfg["config"], split=cfg["split"])
            
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

            global_id = 0
            for row in ds.select(range(min(500, len(ds)))):
                docs = normalize_document(row, name, cfg["field"])
                
                if not docs:
                    continue

                embeddings = model.encode(docs, convert_to_tensor=False)
                
                points = [
                    PointStruct(
                        id=global_id + j,
                        vector=embeddings[j].tolist(),
                        payload={"text": docs[j], "benchmark": name}
                    )
                    for j in range(len(docs))
                ]
                client.upsert(collection_name=collection_name, points=points)
                global_id += len(docs)
            
            print(f"✅ {collection_name} is ready with {global_id} vectors.")
        except Exception as e:
            print(f"❌ Failed {name}: {e}")

if __name__ == "__main__":
    ingest_all()