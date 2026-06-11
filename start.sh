#!/usr/bin/env bash

# Define color codes for clean CLI outputs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0;0m' # No Color

echo -e "${BLUE}=== Starting ATR Model Production Workbench ===${NC}"

# Check if .env has valid keys
if grep -q "GEMINI_API_KEY=YOUR_" .env || grep -q "GOOGLE_API_KEY=YOUR_" .env; then
    echo -e "${YELLOW}[!] Warning: API keys in your .env file might still be placeholders. Configure them if LLM features fail.${NC}"
fi

# Check if ports 8000 or 5173 are already in use, and free them if so
echo -e "${BLUE}[~] Checking port availability (8000 & 5173)...${NC}"
for PORT in 8000 5173; do
    PID=$(lsof -i :$PORT -t 2>/dev/null || true)
    if [ ! -z "$PID" ]; then
        echo -e "${YELLOW}[!] Port $PORT is already in use by process $PID. Terminating process...${NC}"
        kill -9 $PID 2>/dev/null || true
        # Wait a moment for the socket to release
        sleep 0.5
    fi
done

# 1. Start backend FastAPI server
echo -e "${BLUE}[~] Activating backend venv and launching FastAPI server...${NC}"
cd backend
source venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!
deactivate
cd ..

# 2. Start frontend dev server
echo -e "${BLUE}[~] Launching frontend Vite development server...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# 3. Handle termination signal and cleanup children processes
cleanup() {
    echo -e "\n${YELLOW}[~] Terminating workbench servers...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo -e "${GREEN}[+] Stopped all servers. Goodbye!${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Keep script running to print stdout logs
wait
