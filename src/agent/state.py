from typing import TypedDict, Optional

class AgentState(TypedDict):
    api_key: Optional[str]
    has_fhir: bool
    fhir_resource_type: Optional[str]
    fhir_resource_id: Optional[str]
    fhir_output: Optional[str]
    query: str
    has_drug_query: bool
    drug_names: Optional[list[str]]
    drug_labels: Optional[list[dict]]
    search_query: Optional[str]
    abstracts: list[dict]
    llm_response: Optional[str]
    claims: Optional[list[str]]
    scored_claims: Optional[list[dict]]
    confidence_score: Optional[float]
    final_response: Optional[dict]