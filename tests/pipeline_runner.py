from src.agent.graph import agent
import sys

def main():
    result = agent.invoke({
        "query": "What are the first line treatments for atrial fibrillation?",
        "cache_hit": False,
        "abstracts": [],
        "llm_response": None,
        "claims": None,
        "scored_claims": None,
        "confidence_score": None,
        "final_response": None
    })

    print(result["final_response"])

if __name__ == "__main__":
    sys.exit(main())