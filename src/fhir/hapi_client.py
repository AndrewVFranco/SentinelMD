import requests

def fetch_resource(resource_type: str, resource_id: str) -> dict:
    base_url = "https://hapi.fhir.org/"
    try:
        response = requests.get(f'{base_url}baseR4/{resource_type}/{resource_id}')
        response.raise_for_status()
        if response.status_code == 200 and response.json().get("resourceType"):
            return response.json()
    except Exception as e:
        print(f"FHIR lookup failed: {e}")
        return None