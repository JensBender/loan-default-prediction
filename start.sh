#!/bin/sh

# Start the FastAPI backend with Uvicorn 
uvicorn backend.app:app --host 0.0.0.0 --port 8000 &

# Start the Gradio frontend 
python -m frontend.app
