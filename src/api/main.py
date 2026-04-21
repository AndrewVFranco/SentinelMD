from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.agent.graph import agent
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional
import json
import asyncio

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
    allow_origins=[
        "http://localhost:3000",
        "127.0.0.1:3000",
        "https://andrewvfranco-sentinelmd.hf.space",
        "https://*.hf.space",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/query")
async def query(request: QueryRequest):
    initial_state = {
        "api_key": request.api_key,
        "query": request.query,
        "search_query": None,
        "abstracts": [],
        "llm_response": None,
        "claims": None,
        "scored_claims": None,
        "confidence_score": None,
        "final_response": None,
        "has_drug_query": False,
        "drug_names": None,
        "drug_labels": None,
        "has_fhir": request.has_fhir,
        "fhir_resource_type": request.fhir_resource_type,
        "fhir_resource_id": request.fhir_resource_id,
        "fhir_output": None
    }

    async def event_generator():
        try:
            async for output in agent.astream(initial_state):
                node_name = list(output.keys())[0]
                yield f"data: {json.dumps({'status': node_name})}\n\n"
                await asyncio.sleep(0.01)

                if node_name == "assembly":
                    final_state = output["assembly"]
                    yield f"data: {json.dumps(final_state)}\n\n"

        except Exception as e:
            print(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/fhir")
async def fhir(request: QueryRequest):
    initial_state = {
        "api_key": request.api_key,
        "query": request.query,
        "search_query": None,
        "abstracts": [],
        "llm_response": None,
        "claims": None,
        "scored_claims": None,
        "confidence_score": None,
        "final_response": None,
        "has_drug_query": False,
        "drug_names": None,
        "drug_labels": None,
        "has_fhir": True,  # Hardcoded to True for this endpoint
        "fhir_resource_type": request.fhir_resource_type,
        "fhir_resource_id": request.fhir_resource_id,
        "fhir_output": None
    }

    async def event_generator():
        try:
            async for output in agent.astream(initial_state):
                node_name = list(output.keys())[0]
                yield f"data: {json.dumps({'status': node_name})}\n\n"
                await asyncio.sleep(0.01)

                if node_name == "assembly":
                    final_state = output["assembly"]
                    yield f"data: {json.dumps(final_state)}\n\n"

        except Exception as e:
            print(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}