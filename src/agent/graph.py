from langgraph.graph import StateGraph
from src.agent.nodes import check_cache, pubmed_retrieval, llm_generation, parse_claims, nli_scoring, \
    confidence_scoring, assembly, route_after_cache, preprocess_query
from src.agent.state import AgentState

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("check_cache", check_cache)
graph.add_node("preprocess_query", preprocess_query)
graph.add_node("pubmed_retrieval", pubmed_retrieval)
graph.add_node("llm_generation", llm_generation)
graph.add_node("parse_claims", parse_claims)
graph.add_node("nli_scoring", nli_scoring)
graph.add_node("confidence_scoring", confidence_scoring)
graph.add_node("assembly", assembly)

# Conditional edge
graph.add_conditional_edges(
    "check_cache",
    route_after_cache,
    {"pubmed_retrieval": "pubmed_retrieval", "llm_generation": "llm_generation"}
)

# Add edges
graph.add_edge("preprocess_query", "check_cache")
graph.add_edge("pubmed_retrieval", "llm_generation")
graph.add_edge("llm_generation", "parse_claims")
graph.add_edge("parse_claims", "nli_scoring")
graph.add_edge("nli_scoring", "confidence_scoring")
graph.add_edge("confidence_scoring", "assembly")

# Set entry and finish
graph.set_entry_point("preprocess_query")
graph.set_finish_point("assembly")

# Compile
agent = graph.compile()