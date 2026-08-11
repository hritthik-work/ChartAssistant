from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

PATIENT_SHORTHAND = re.compile(
    r"\b(?:patient\s*[-#]?\s*|syn(?:thetic)?[ -]?p(?:atient)?[ -]?)(\d{1,3})\b",
    re.IGNORECASE,
)
CANONICAL_PATIENT = re.compile(r"\bSYN-P\d{3}\b", re.IGNORECASE)
HARM_ACTION = re.compile(
    r"\b(kill|murder|poison|harm|hurt|overdose|suffocate|strangle)\b", re.IGNORECASE
)
ENABLEMENT = re.compile(
    r"\b(how\s+(?:can|do|would)\s+i|tell\s+me\s+how|ways?\s+to|"
    r"can\s+i|should\s+i|would\b.{0,80}\bwork)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryPolicyDecision:
    original_query: str
    normalized_query: str
    route: Literal["rag", "safety_refusal"]


def normalize_query(query: str) -> str:
    collapsed = " ".join(query.split())

    def replace(match: re.Match[str]) -> str:
        return f"SYN-P{int(match.group(1)):03d}"

    return PATIENT_SHORTHAND.sub(replace, collapsed)


def patient_ids(query: str) -> list[str]:
    return sorted({item.upper() for item in CANONICAL_PATIENT.findall(query)})


def evaluate_query_policy(query: str) -> QueryPolicyDecision:
    normalized = normalize_query(query)
    route: Literal["rag", "safety_refusal"] = "rag"
    if HARM_ACTION.search(normalized) and ENABLEMENT.search(normalized):
        route = "safety_refusal"
    return QueryPolicyDecision(query, normalized, route)
