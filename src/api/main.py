from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from src.agent.graph import agent
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

@asynccontextmanager
async def lifespan(app):
    yield
app = FastAPI(lifespan=lifespan)

@app.post("/query")
async def query(request: QueryRequest):
    try:
        result = agent.invoke({
            "query": request.query,
            "search_query": None,
            "cache_hit": False,
            "abstracts": [],
            "llm_response": None,
            "claims": None,
            "scored_claims": None,
            "confidence_score": None,
            "final_response": None
        })

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok"}