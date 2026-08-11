from __future__ import annotations

from typing import Literal


class AppError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        category: Literal[
            "validation",
            "configuration",
            "index_schema_mismatch",
            "provider_timeout",
            "rate_limit",
            "provider_error",
            "invalid_model_output",
            "internal_error",
        ],
        status_code: int,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        self.retryable = retryable


class ValidationAppError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, category="validation", status_code=422)


class ConfigurationAppError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, category="configuration", status_code=503)


class ProviderAppError(AppError):
    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(
            message,
            category="provider_error",
            status_code=503 if retryable else 502,
            retryable=retryable,
        )
