import mlflow
from src.core.config import settings

def log_query_run(final_response: dict) -> None:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("SentinelMD")
    supported_count = len([c for c in final_response["scored_claims"] if c["label"] == "Supported"])
    unverifiable_count = len([c for c in final_response["scored_claims"] if c["label"] == "Unverifiable"])
    contradicted_count = len([c for c in final_response["scored_claims"] if c["label"] == "Contradicted"])

    with mlflow.start_run():
        mlflow.log_param("query", final_response["query"])
        mlflow.log_metric("abstracts_retrieved_count", len(final_response["abstracts"]))
        mlflow.log_metric("confidence_score", final_response['confidence_score'])
        mlflow.log_metric("supported_claims", supported_count)
        mlflow.log_metric("unverifiable_claims", unverifiable_count)
        mlflow.log_metric("contradicted_claims", contradicted_count)
        mlflow.log_metric("total_claims", len(final_response["scored_claims"]))
