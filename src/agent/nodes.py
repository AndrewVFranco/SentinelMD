import numpy as np
from src.agent.state import AgentState
from src.retrieval.cache import get_cache, set_cache
from src.retrieval.vector_store import add_abstracts, query_abstracts
from src.retrieval.pubmed import search_pubmed
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from sentence_transformers import CrossEncoder
from src.core.config import settings

_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=settings.GEMINI_API_KEY)
_nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")

def check_cache(state: AgentState):
    cached_result = get_cache(state["query"])
    if cached_result:
        return {"cache_hit": True, "abstracts": cached_result}
    else:
        return {"cache_hit": False}

def route_after_cache(state: AgentState) -> str:
    if state["cache_hit"]:
        return "llm_generation"
    return "pubmed_retrieval"

def pubmed_retrieval(state: AgentState):
    if not state["cache_hit"]:
        results = search_pubmed(state["query"])
        add_abstracts(results)
        abstracts = query_abstracts(state["query"])
        set_cache(state["query"], abstracts)
        return {"abstracts": abstracts}

def llm_generation(state: AgentState):
    context = "\n\n".join([f"Title: {a['title']}\nAbstract: {a['abstract']}"
                           for a in state["abstracts"]])

    prompt = f"""You are a clinical assistant in charge of extracting insights from medical literature. Use the following documentation to answer the query.

    Literature:
    {context}

    Query: {state["query"]}

    Provide a detailed clinical response based solely on the provided literature."""

    response = _llm.invoke(prompt)
    return {"llm_response": response.content}

def parse_claims(state: AgentState):
    parser = JsonOutputParser()

    prompt = f"""Extract all discrete factual claims from the following clinical response.
    Return ONLY a JSON array of strings, no other text.
    Each claim should be a single verifiable factual statement.

    Response:
    {state["llm_response"]}

    Return format: ["claim 1", "claim 2", "claim 3"]"""

    response = _llm.invoke(prompt)
    claims = parser.parse(response.content)
    return {"claims": claims}

def nli_scoring(state: AgentState):
    scored_claims = []
    labels = ["Contradicted", "Supported", "Unverifiable"]
    for claim in state["claims"]:
        best_score = -1
        best_result = None

        for abstract in state["abstracts"]:
            scores = _nli_model.predict([(abstract["abstract"], claim)])[0]
            label_idx = int(np.argmax(scores))
            if scores[label_idx] > best_score:
                best_score = scores[label_idx]
                best_result = {
                    "claim": claim,
                    "label": labels[label_idx],
                    "score": float(best_score),
                    "evidence": abstract["abstract"]
                }

        scored_claims.append(best_result)

    return {"scored_claims": scored_claims}

def confidence_scoring(state: AgentState):
    weights = {"Supported": 1.0, "Unverifiable": 0.5, "Contradicted": 0.0}
    score = np.mean([weights[claim["label"]] for claim in state["scored_claims"]])
    return {"confidence_score": score}

def assembly(state: AgentState):
    return {"final_response": {
        "query": state["query"],
        "response": state["llm_response"],
        "confidence_score": state["confidence_score"],
        "scored_claims": state["scored_claims"],
        "abstracts": state["abstracts"]
        }
    }
