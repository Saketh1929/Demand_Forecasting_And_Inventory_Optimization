#!/bin/bash

# Render exposes one public port per service. In the combined image, Streamlit
# owns that port and FastAPI listens on an internal port.
APP_PORT=${PORT:-8501}
BACKEND_PORT=${BACKEND_PORT:-8000}

PORT="$BACKEND_PORT" uvicorn backend.main:app --host 0.0.0.0 --port "$BACKEND_PORT" &

# Wait a moment for the backend to initialize
sleep 2

# Both processes run in this container, so the frontend reaches FastAPI through
# the container loopback interface.
export DFIO_API_BASE_URL="http://127.0.0.1:${BACKEND_PORT}"

# Start Streamlit with explicit production settings
# Headless mode prevents it from trying to open a browser window
# CORS/XSRF are disabled because Render's edge proxy handles external routing.
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

exec streamlit run Frontend/app.py --server.port "$APP_PORT" --server.address 0.0.0.0
