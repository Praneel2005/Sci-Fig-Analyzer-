#!/bin/bash
echo "Starting Backend (FastAPI)..."
cd /home/drive4/FigCapsHF/backend
# Start backend in background
uvicorn app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Starting Frontend (Vite)..."
cd /home/drive4/FigCapsHF/frontend
# Start frontend
npm run dev

# Trap ctrl-c and kill background process
trap "kill $BACKEND_PID" EXIT
