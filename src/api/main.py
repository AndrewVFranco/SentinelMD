from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from src.agent.graph import agent
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

class QueryRequest(BaseModel):
    query: str = ""
    api_key: Optional[str] = None
    fhir_resource_type: Optional[str] = None
    fhir_resource_id: Optional[str] = None
    has_fhir: bool = False

@asynccontextmanager
async def lifespan(app):
    yield
app = FastAPI(lifespan=lifespan, title="SentinelMD")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query")
async def query(request: QueryRequest):
    try:
        result = agent.invoke({
            "query": request.query,
            "search_query": None,
            "abstracts": [],
            "llm_response": None,
            "claims": None,
            "scored_claims": None,
            "confidence_score": None,
            "final_response": None,
            "drug_names": None,
            "drug_labels": None,
            "has_fhir": request.has_fhir,
            "fhir_resource_type": request.fhir_resource_type,
            "fhir_resource_id": request.fhir_resource_id,
            "fhir_output": None
        })

        return result
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/fhir")
async def fhir(request: QueryRequest):
    try:
        result = agent.invoke({
            "query": request.query,
            "search_query": None,
            "abstracts": [],
            "llm_response": None,
            "claims": None,
            "scored_claims": None,
            "confidence_score": None,
            "final_response": None,
            "drug_names": None,
            "drug_labels": None,
            "has_fhir": True,
            "fhir_resource_type": request.fhir_resource_type,
            "fhir_resource_id": request.fhir_resource_id,
            "fhir_output": None
        })
        return result
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"FHIR inference failed: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok"}