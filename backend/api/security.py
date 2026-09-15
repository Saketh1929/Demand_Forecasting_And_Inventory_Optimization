import json
from time import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import Request
from fastapi.responses import JSONResponse

from backend.config import ENABLE_AUTH, API_KEY, ALLOWED_ROLES, PROTECTED_ENDPOINT_POLICIES
from backend.api.schemas import GenericErrorResponse

from collections import defaultdict

request_timestamps = defaultdict(list)
REQUEST_LIMIT_PER_MINUTE = 60
REQUIRE_ROLE_AUTH = True

def generate_request_id() -> str:
    current_time = datetime.now(timezone.utc)
    return f"fcst_{current_time.strftime('%Y%m%d%H%M%S')}_{abs(hash(current_time.microsecond)) % 10000:04d}"

def build_error_response(code: str, message: str, fields: Dict[str, str], request_id: Optional[str] = None):
    return {
        "status": "failed",
        "error": {
            "code": code,
            "message": message,
            "fields": fields,
        },
        "request_id": request_id,
    }

def get_required_roles_for_request(request: Request) -> Optional[set]:
    for path_prefix, policy in PROTECTED_ENDPOINT_POLICIES.items():
        if request.url.path.startswith(path_prefix):
            if request.method not in policy.get("methods", set()):
                return None
            return policy.get("roles", set())
    return None

async def enforce_security_and_rate_limits(request: Request, call_next):
    now = time()
    client_ip = request.client.host if request.client else "unknown"
    request_timestamps[client_ip] = [ts for ts in request_timestamps[client_ip] if now - ts < 60]
    if len(request_timestamps[client_ip]) >= REQUEST_LIMIT_PER_MINUTE:
        return JSONResponse(
            status_code=429,
            content={
                "status": "failed",
                "error": {"code": "RATE_LIMIT_EXCEEDED", "message": "Too many requests. Please retry later.", "fields": {}},
                "request_id": generate_request_id(),
            },
        )
    request_timestamps[client_ip].append(now)

    if not ENABLE_AUTH:
        return await call_next(request)

    if request.url.path.startswith("/docs") or request.url.path.startswith("/openapi") or request.url.path.startswith("/redoc"):
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {API_KEY}":
        return JSONResponse(
            status_code=401,
            content={
                "status": "failed",
                "error": {"code": "UNAUTHORIZED", "message": "Missing or invalid API key.", "fields": {}},
                "request_id": generate_request_id(),
            },
        )

    required_roles = get_required_roles_for_request(request)
    if REQUIRE_ROLE_AUTH and required_roles is not None:
        user_role = request.headers.get("X-User-Role", "").lower()
        if user_role not in ALLOWED_ROLES or user_role not in required_roles:
            return JSONResponse(
                status_code=403,
                content={
                    "status": "failed",
                    "error": {"code": "FORBIDDEN", "message": "You do not have permission to perform this action.", "fields": {}},
                    "request_id": generate_request_id(),
                },
            )
    return await call_next(request)
