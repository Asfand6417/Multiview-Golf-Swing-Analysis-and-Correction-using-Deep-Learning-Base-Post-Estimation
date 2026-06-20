#!/bin/bash

echo "Creating Docker network..."
docker network create golfpose_net || true

echo "Starting Database and Redis..."
docker run -d --name db --network golfpose_net -e POSTGRES_USER=user -e POSTGRES_PASSWORD=password -e POSTGRES_DB=golfpose -p 5432:5432 postgres:15
docker run -d --name redis --network golfpose_net -p 6379:6379 redis:alpine

echo "Building and starting Backend..."
docker build -t golfpose-backend ./backend
docker run -d --name backend --network golfpose_net -p 8000:8000 golfpose-backend

echo "Building and starting Celery Worker..."
docker run -d --name celery_worker --network golfpose_net golfpose-backend celery -A tasks.analysis worker --loglevel=info

echo "Building and starting Frontend..."
docker build -t golfpose-frontend ./frontend
docker run -d --name frontend --network golfpose_net -p 3000:3000 golfpose-frontend

echo "======================================"
echo "All services started successfully!"
echo "Frontend UI:   http://localhost:3000"
echo "Backend API:   http://localhost:8000"
echo "======================================"
