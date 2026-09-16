from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request

from backend.api.routes import router as api_router
from backend.api import security as security_module
from backend.api.security import build_error_response, generate_request_id
from backend.config import ENABLE_AUTH, API_KEY

REQUIRE_ROLE_AUTH = security_module.REQUIRE_ROLE_AUTH


async def enforce_security_and_rate_limits(request: Request, call_next):
    """Apply app-level security settings before delegating to the middleware."""
    security_module.ENABLE_AUTH = ENABLE_AUTH
    security_module.API_KEY = API_KEY
    security_module.REQUIRE_ROLE_AUTH = REQUIRE_ROLE_AUTH
    return await security_module.enforce_security_and_rate_limits(request, call_next)

app = FastAPI(
    title="Demand Forecasting & Inventory Optimization Agent API",
    description="FastAPI production backend providing ML demand prediction, real-time inventory gap analysis, Gemini LLM rationale, and human-in-the-loop audit approvals.",
    version="1.0.0"
)

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(enforce_security_and_rate_limits)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    field_errors = {}
    for error in exc.errors():
        loc = error.get("loc", [])
        field_name = loc[-1] if len(loc) > 1 else "request"
        field_errors[field_name] = error.get("msg", "Invalid value.")

    return JSONResponse(
        status_code=422,
        content=build_error_response(
            "VALIDATION_ERROR",
            "One or more input values are invalid.",
            field_errors,
            request_id=generate_request_id(),
        ),
    )

app.include_router(api_router)
