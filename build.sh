#!/usr/bin/env bash
set -euo pipefail

echo "== Pro Swing AI: Build helper =="
echo "1) Building frontend..."
if [ -d "frontend" ]; then
  (cd frontend && npm install --no-audit --no-fund)
  (cd frontend && npm run build)
  echo "✅ Frontend build complete: frontend/.next (or .output) created"
else
  echo "⚠️ frontend folder not found. Skipping frontend build."
fi

echo ""
echo "2) Backend: Python FastAPI service"
echo "To start the backend (local dev) run:"
echo "   python \"main_SKELETON_3D(2).py\""
echo "or run (if you prefer uvicorn and have a module entry):"
echo "   uvicorn main_SKELETON_3D(2):api --reload --port 8000"
echo ""
echo "If you want to dockerize, create Dockerfiles for frontend/backend and a docker-compose.yml."
echo "Build finished."
