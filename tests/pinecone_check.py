from src.retrieval.pubmed import search_pubmed
from src.retrieval.vector_store import add_abstracts, query_abstracts
import sys

def main():
    search_results = search_pubmed("pulmonary embolism", max_results=5)
    add_abstracts(search_results)
    results = query_abstracts("chest pain treatment")
    print(results)

if __name__ == "__main__":
    sys.exit(main())