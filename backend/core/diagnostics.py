from __future__ import annotations

import json
import logging
from contextvars import ContextVar, Token
from typing import Any

REQUEST_ID: ContextVar[str | None] = ContextVar("healthchat_request_id", default=None)
MAX_MESSAGE_LENGTH = 2_000
BASE_LOGGER = logging.getLogger("healthchat")
BASE_LOGGER.addHandler(logging.NullHandler())
BASE_LOGGER.propagate = False


def configure_logging(level: str) -> logging.Logger:
    logger = BASE_LOGGER
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not any(not isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        logger.handlers = [
            handler for handler in logger.handlers if not isinstance(handler, logging.NullHandler)
        ]
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)
    return logger


def bind_request_id(request_id: str) -> Token[str | None]:
    return REQUEST_ID.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    REQUEST_ID.reset(token)


def current_request_id() -> str | None:
    return REQUEST_ID.get()


def _limited(value: Any) -> str:
    rendered = str(value)
    if len(rendered) <= MAX_MESSAGE_LENGTH:
        return rendered
    return f"{rendered[:MAX_MESSAGE_LENGTH]}...<truncated>"


def exception_chain(exc: BaseException, *, limit: int = 6) -> list[dict[str, Any]]:
    """Return safe provider diagnostics without headers, credentials, or request bodies."""
    chain: list[dict[str, Any]] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and len(chain) < limit and id(current) not in seen:
        seen.add(id(current))
        detail: dict[str, Any] = {
            "type": type(current).__name__,
            "message": _limited(current),
        }
        for attribute in ("status_code", "code", "request_id"):
            value = getattr(current, attribute, None)
            if value not in (None, ""):
                detail[attribute] = value

        provider_error = getattr(current, "error", None)
        if provider_error is not None:
            for attribute in ("code", "message"):
                value = getattr(provider_error, attribute, None)
                if value not in (None, ""):
                    detail[f"provider_{attribute}"] = _limited(value)

        response = getattr(current, "response", None)
        headers = getattr(response, "headers", None) or {}
        for header, field in (
            ("request-id", "service_request_id"),
            ("x-ms-request-id", "service_request_id"),
            ("x-ms-client-request-id", "client_request_id"),
        ):
            value = headers.get(header)
            if value and field not in detail:
                detail[field] = value

        body = getattr(current, "body", None)
        if isinstance(body, dict):
            for field in ("type", "param", "code", "message"):
                value = body.get(field)
                if value not in (None, ""):
                    detail[f"provider_{field}"] = _limited(value)

        chain.append(detail)
        current = current.__cause__ or current.__context__
    return chain


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    *,
    error: BaseException | None = None,
    include_traceback: bool = False,
    **fields: Any,
) -> None:
    payload = {
        "event": event,
        "request_id": current_request_id(),
        **{key: value for key, value in fields.items() if value is not None},
    }
    if error is not None:
        payload["exception_chain"] = exception_chain(error)
    logger.log(
        level,
        json.dumps(payload, default=str, sort_keys=True),
        exc_info=error if include_traceback else None,
    )
