#!/usr/bin/env bash

# Upgrade pip, setuptools, and wheel FIRST
pip install --upgrade pip setuptools wheel

# Then install your full requirements
pip install -r requirements.txt

# Start your FastAPI app using uvicorn
exec uvicorn app:app --host=0.0.0.0 --port=$PORT