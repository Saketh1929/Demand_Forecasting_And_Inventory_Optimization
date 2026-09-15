#!/bin/bash

# 1. Capture Railway's assigned port before unsetting it
RAILWAY_PORT=${PORT:-8080}

# Unset PORT so uvicorn doesn't accidentally pick it up and conflict with Streamlit
unset PORT

# 2. Start the FastAPI backend in the background on a fixed internal port (8000)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Wait a moment for the backend to initialize
sleep 2

# 3. Tell the Frontend to communicate with the local backend (using 127.0.0.1 to avoid Docker IPv6 localhost resolution issues)
export DFIO_API_BASE_URL="http://127.0.0.1:8000"

# 4. Start Streamlit with explicit production settings
# Headless mode prevents it from trying to open a browser window
# CORS/XSRF are disabled because Railway's edge proxy handles external routing securely
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

streamlit run Frontend/app.py --server.port $RAILWAY_PORT --server.address 0.0.0.0
