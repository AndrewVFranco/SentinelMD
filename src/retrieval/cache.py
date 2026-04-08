import redis
import json
from src.core.config import settings

_client = None

def get_client():
    global _client
    if _client is None:
        _client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)
    return _client

def get_cache(key: str):
    value = get_client().get(key)
    if value is None:
        return None
    return json.loads(value)

def set_cache(key: str, value: list[dict]):
    get_client().setex(key, 86400, json.dumps(value))  # 86400 = 24 hours
