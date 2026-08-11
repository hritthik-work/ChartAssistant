# RAG flow

```mermaid
flowchart TD
    Query["Natural-language question"] --> Normalize["Normalize whitespace and patient shortcuts"]
    Normalize --> Safety{"Deterministic safety refusal?"}
    Safety -- Yes --> SafeStop["Refuse with zero model and retrieval calls"]
    Safety -- No --> Resolve["Match exact patient ID, name, chart number, or unique number"]
    Resolve --> Unique{"Unique patient or general guide/policy request?"}
    Unique -- No --> Clarify["Request patient clarification"]
    Unique -- Yes --> Agent["MAF agent with system prompt v5"]
    Agent --> Retrieve["retrieve_evidence exactly once"]
    Retrieve --> Search["Azure hybrid search with patient/source filter"]
    Search --> Bound["Threshold, lexical-support, role, and context-budget checks"]
    Bound --> Lookup{"Explicit ICD request?"}
    Lookup -- Yes --> ICD["mock_icd10_lookup exactly once"]
    Lookup -- No --> Candidate["Strict ModelCandidate"]
    ICD --> Candidate
    Candidate --> Ground["Validate exact quotes, IDs, patient, status, and tool result"]
    Ground --> Valid{"Valid?"}
    Valid -- Yes --> Response["QueryResponse with sources, confidence, gaps, and trace"]
    Valid -- No --> Repair["One repair call with no tools"]
    Repair --> Recheck{"Grounded now?"}
    Recheck -- Yes --> Response
    Recheck -- No --> Fail["Typed invalid-model-output error"]
```

Patient resolution happens before the agent or search. Canonical references such as `SYN-P004`,
shortcuts such as “patient 4,” exact stored names, and unique numeric identifiers are supported.
Fuzzy guessing is not used. General guide or policy questions may search the matching source role
without a patient.

No matching evidence is a valid outcome. The assistant returns partial or refused with missing
information instead of answering from model memory. Azure filters are checked again after retrieval,
and fallback backend use is recorded in the response trace.
