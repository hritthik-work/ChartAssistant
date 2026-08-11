# Ingestion flow

```mermaid
flowchart TD
    Files["Patient reference and synthetic files"] --> Validate["Validate confirmation, count, size, and type"]
    Validate --> Metadata{"Structured metadata present?"}
    Metadata -- Yes --> Match{"Entered and embedded patient match?"}
    Match -- No --> Reject["Typed validation error"]
    Match -- Yes --> Parse["Parse document"]
    Metadata -- No --> Assign["Assign entered patient reference"]
    Assign --> Parse
    Parse --> Sections["Preserve pages, headings, or paragraph sections"]
    Sections --> Chunk["Create section-aware chunks with overlap"]
    Chunk --> Hash["Calculate stable IDs and content hashes"]
    Hash --> Compare["Read existing hashes from Azure AI Search"]
    Compare --> Changed{"Changed chunks?"}
    Changed -- Yes --> Embed["Embed changed chunks only"]
    Embed --> Update["Upsert changed and delete stale chunks"]
    Changed -- No --> Skip["Mark document unchanged"]
    Update --> Save["Atomically save normalized text, chunks, and manifest"]
    Skip --> Save
    Save --> Refresh["Refresh local retrieval and knowledge-base counts"]
    Refresh --> Complete["Ready for questions"]
```

The browser uses `POST /ingestion/jobs` and polls `GET /ingestion/jobs/{job_id}`. Reported stages
are queued, validating, parsing, chunking, checking changes, embedding, updating the index, saving,
completed, or failed. The synchronous `POST /ingestion` endpoint remains available for structured
sample documents and command-line use.

Generic-chart IDs come from the normalized patient reference and filename; structured sample files
retain their embedded document IDs. Stable chunk hashes make repeat uploads idempotent. Raw uploaded
bytes and vectors are not persisted locally. Text-based PDFs are supported; encrypted and image-only
PDFs return a clear OCR-not-supported error.
