def parse_fhir_resource(resource: dict) -> str:
    resource_type = resource.get("resourceType")

    if resource_type == "Condition":
        condition = (
            resource.get("code", {}).get("text") or
            resource.get("code", {}).get("coding", [{}])[0].get("display")
        )
        if condition:
            return f"What are the treatment options and management guidelines for {condition}?"

    elif resource_type == "MedicationRequest":
        med = (
            resource.get("medicationCodeableConcept", {}).get("text") or
            resource.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("display")
        )
        if med:
            return f"What are the clinical indications, risks, and side effects of {med}?"

    elif resource_type == "DiagnosticReport":
        finding = (
            resource.get("conclusion") or
            resource.get("code", {}).get("text")
        )
        if finding:
            return f"What are the clinical implications and management options for {finding}?"

    return None