from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # NCBI
    NCBI_API_KEY: str

    # Hugging Face
    HF_TOKEN: str

    # Ollama
    OLLAMA_SERVER_IP: str
    OLLAMA_SERVER_PORT: int = 11434
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_MAX_TOKENS: int = 1024
    OLLAMA_TEMPERATURE: float = 0.1

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str

    # Redis
    REDIS_HOST: str
    REDIS_PORT: str

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_ARTIFACT_LOCATION: str = "logs/mlflow"

    # FastAPI
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    ENV: str = "development"
    DEBUG: bool = True

    class Config:
        env_file = Path(__file__).resolve().parent.parent.parent / ".env"

settings = Settings()