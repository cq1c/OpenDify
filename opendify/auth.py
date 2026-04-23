"""API Key 校验。"""

from fastapi import Request

from .config import AUTH_MODE, VALID_API_KEYS
from .errors import APIError


async def verify_api_key(request: Request) -> str:
    if AUTH_MODE == "disabled":
        return ""
    auth = request.headers.get("Authorization") or ""
    key = auth[7:] if auth.lower().startswith("bearer ") else auth.strip()
    if not key:
        key = (request.headers.get("X-API-Key") or "").strip()
    if not key or key not in VALID_API_KEYS:
        raise APIError(
            401,
            "Invalid API key",
            error_type="invalid_request_error",
            code="invalid_api_key",
        )
    return key
