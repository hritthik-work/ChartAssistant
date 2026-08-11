# User flow

```mermaid
flowchart TD
    Open["Open Chart Assistant"] --> Status["Automatic system health check"]
    Status --> Ask["Enter a natural-language chart question"]
    Ask --> Patient{"Unique patient or general guide/policy request?"}
    Patient -- No --> Clarify["Show a friendly clarification"]
    Clarify --> Ask
    Patient -- Yes --> Loading["Clear old result and show current stage"]
    Loading --> Result["Show answer and resolved patient"]
    Result --> Sources["Show friendly source cards"]
    Sources --> Details["Optional confidence, gaps, tools, and trace"]

    Open --> Add["Add patient charts"]
    Add --> Select["Enter patient reference and select synthetic files"]
    Select --> Progress["Show upload and server-processing timeline"]
    Progress --> Ready["Refresh patient and document counts"]
    Ready --> Ask
```

Question requests carry a request identity in the browser. Cancelling or starting another question
prevents an older response from replacing the current result. Missing, unknown, or ambiguous patient
context stops before retrieval and model calls.

The upload form accepts PDF, DOCX, TXT, and Markdown. It shows selected files, removal controls,
actual browser upload progress, server processing stages, elapsed time, and safe retryable errors.
