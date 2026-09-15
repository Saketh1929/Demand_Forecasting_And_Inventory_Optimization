#!/bin/bash

# 1. Start the FastAPI backend in the background on a fixed internal port (8000)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# 2. Tell the Frontend to communicate with the local backend
export DFIO_API_BASE_URL="http://localhost:8000"

# 3. Start the Streamlit frontend in the foreground using Railway's assigned $PORT
streamlit run Frontend/app.py --server.port $PORT --server.address 0.0.0.0
