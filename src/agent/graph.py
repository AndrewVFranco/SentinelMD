from langgraph.graph import StateGraph, START
from src.agent.nodes import pubmed_retrieval, llm_generation, parse_claims, nli_scoring, \
    confidence_scoring, assembly, preprocess_query, detect_medications, fda_enrichment, \
    route_after_medication_detection, fhir_input, route_entry
from src.agent.state import AgentState

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("fhir_input", fhir_input)
graph.add_node("preprocess_query", preprocess_query)
graph.add_node("pubmed_retrieval", pubmed_retrieval)
graph.add_node("llm_generation", llm_generation)
graph.add_node("detect_medications", detect_medications)
graph.add_node("fda_enrichment", fda_enrichment)
graph.add_node("parse_claims", parse_claims)
graph.add_node("nli_scoring", nli_scoring)
graph.add_node("confidence_scoring", confidence_scoring)
graph.add_node("assembly", assembly)

# Conditional edge
graph.add_conditional_edges(
    "detect_medications",
    route_after_medication_detection,
    {"fda_enrichment": "fda_enrichment", "llm_generation": "llm_generation"}
)

# Add edges
graph.add_edge("fhir_input", "preprocess_query")
graph.add_edge("preprocess_query", "pubmed_retrieval")
graph.add_edge("pubmed_retrieval", "detect_medications")
graph.add_edge("fda_enrichment", "llm_generation")
graph.add_edge("llm_generation", "parse_claims")
graph.add_edge("parse_claims", "nli_scoring")
graph.add_edge("nli_scoring", "confidence_scoring")
graph.add_edge("confidence_scoring", "assembly")

# Set entry and finish
graph.add_conditional_edges(
    START,
    route_entry,
    {"fhir_input": "fhir_input", "preprocess_query": "preprocess_query"}
)

graph.set_finish_point("assembly")

# Compile
agent = graph.compile()