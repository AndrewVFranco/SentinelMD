from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from src.core.config import settings
import os
from src.retrieval.pubmed import search_pubmed
import sys

os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.HF_TOKEN
model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
_collection = None

def _embed_text(text: str) -> list[float]:
    embedding = model.encode(text)
    embedding = embedding.tolist()
    return embedding


def get_collection():
    global _collection
    if _collection is None:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        _collection = pc.Index(settings.PINECONE_INDEX_NAME)
    return _collection


def add_abstracts(abstracts: list[dict]):
    data_list = []

    for item in abstracts:
        data = {'id': item["pmid"],
                'values': _embed_text(item["abstract"]),
                'metadata': {
                    "title": item["title"],
                    "abstract": item["abstract"],
                    "pmid": item["pmid"]
                    }
                }
        data_list.append(data)

    get_collection().upsert(vectors=data_list)

def query_abstracts(query: str, n_results: int = 5) -> list[dict]:
    embedding = _embed_text(query)
    results = get_collection().query(
        vector=embedding,
        top_k=n_results,
        include_metadata=True
    )
    return results

def main():
    search_results = search_pubmed("myocardial infarction", max_results=5)
    add_abstracts(search_results)
    results = query_abstracts("chest pain treatment")
    print(results)

if __name__ == "__main__":
    sys.exit(main())
