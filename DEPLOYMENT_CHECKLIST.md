# Production Deployment Checklist

## 1. HTTPS and exposure
- Enforce HTTPS at the ingress, reverse proxy, or platform layer before exposing the API publicly.
- Do not rely on application code alone for TLS termination.
- Configure a trusted certificate and redirect all HTTP traffic to HTTPS.
- Use `X-Forwarded-Proto` or equivalent proxy headers only if the platform is configured to terminate TLS.

## 2. Authentication and authorization
- Set `ENABLE_AUTH=true` in the environment for non-local deployments.
- Provide a strong `API_KEY` value for service-to-service access.
- Require `Authorization: Bearer <token>` for protected routes.
- Use `X-User-Role` for user-dependent authorization checks on admin-sensitive routes.
- Keep admin/manager permissions limited to the approval flow and other privileged actions.

## 3. Rate limiting
- Keep request throttling enabled at the middleware level.
- Tune the limit to your expected traffic shape and backend capacity.
- Consider applying an additional reverse-proxy rate limit for burst protection.

## 4. Model governance
- Maintain an approved model registry and version verification before serving predictions.
- Block unsupported or unapproved model versions in production.
- Keep model metadata and validation logs in a central audit trail.

## 5. Input validation and safety
- Keep backend validation authoritative for all inbound payloads.
- Reject unknown store/product/region/category combinations with `404` error responses.
- Avoid exposing stack traces or raw internals in API error responses.

## 6. Observability
- Log request ID, model version, latency, and outcome for every forecast request.
- Track unauthorized, failed, and rate-limited requests separately.
- Monitor API uptime, model availability, and inference latency.

## 7. Operational reliability
- Set timeouts for inbound requests and model inference calls.
- Include health and readiness endpoints for platform monitoring.
- Add graceful degradation for model outages and fallback behavior.

## 8. Suggested deployment pattern
- Public exposure: HTTPS-enabled reverse proxy / API gateway
- App runtime: FastAPI service behind TLS-terminating infrastructure
- Security: mTLS or gateway-level auth for internal services where available
- Monitoring: centralized logs and alerting for unauthorized traffic and model failures

## 9. Render deployment

Deploy one Render Web Service using the root `Dockerfile`.

- Health check path: `/`
- FastAPI's `/api/health` endpoint is available only inside the container because Streamlit owns Render's public port.
- Render supplies the public `PORT` automatically for Streamlit.
- FastAPI listens internally on `BACKEND_PORT` (default `8000`).
- Streamlit connects to FastAPI through `http://127.0.0.1:8000` inside the container.
- Set `GEMINI_API_KEY`, `ENABLE_AUTH`, and `API_KEY` as Render environment variables.
- No `BACKEND_URL` or second Render service is required.
