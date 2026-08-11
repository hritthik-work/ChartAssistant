from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ..models import Chunk
from .policy import patient_ids

REFERENCE_NUMBER = re.compile(
    r"\b(?:patient|chart|member|record|id|number|no\.?|#)\s*[-:#]?\s*(\d{1,12})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PatientResolution:
    status: Literal["resolved", "missing", "unknown", "ambiguous"]
    reference: str | None
    message: str | None = None


def patient_references(chunks: list[Chunk]) -> list[str]:
    return sorted(
        {chunk.patient_id for chunk in chunks if chunk.patient_id},
        key=str.casefold,
    )


def resolve_patient(query: str, references: list[str]) -> PatientResolution:
    canonical = patient_ids(query)
    if canonical:
        if not references:
            return PatientResolution("resolved", canonical[0])
        matches = [
            reference
            for reference in references
            if reference.casefold() in {item.casefold() for item in canonical}
        ]
        if len(matches) == 1:
            return PatientResolution("resolved", matches[0])
        return PatientResolution(
            "unknown",
            None,
            f"No chart was found for {canonical[0]}.",
        )

    if not references:
        return PatientResolution("missing", None, "No patient charts are available yet.")

    exact = [
        reference
        for reference in references
        if re.search(
            rf"(?<![A-Za-z0-9]){re.escape(reference)}(?![A-Za-z0-9])",
            query,
            re.IGNORECASE,
        )
    ]
    if len(exact) == 1:
        return PatientResolution("resolved", exact[0])
    if len(exact) > 1:
        return PatientResolution(
            "ambiguous",
            None,
            "The question refers to more than one patient chart. Please name one patient.",
        )

    number_match = REFERENCE_NUMBER.search(query)
    if number_match:
        requested = number_match.group(1).lstrip("0") or "0"
        numeric_matches = []
        for reference in references:
            suffix = re.search(r"(\d+)\s*$", reference)
            if suffix and (suffix.group(1).lstrip("0") or "0") == requested:
                numeric_matches.append(reference)
        if len(numeric_matches) == 1:
            return PatientResolution("resolved", numeric_matches[0])
        if not numeric_matches:
            return PatientResolution(
                "unknown",
                None,
                f"No chart was found for patient {number_match.group(1)}.",
            )
        return PatientResolution(
            "ambiguous",
            None,
            f"Patient number {number_match.group(1)} matches more than one chart. "
            "Please use the full patient reference.",
        )

    if len(references) == 1:
        return PatientResolution("resolved", references[0])
    return PatientResolution(
        "missing",
        None,
        "Please include a patient ID, name, or chart number in the question.",
    )
