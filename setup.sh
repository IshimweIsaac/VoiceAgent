#!/usr/bin/env bash
# =============================================================================
# VoiceAgent — One-time Setup Script
# Creates a virtual environment, installs dependencies, and configures .env
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== VoiceAgent Setup ==="
echo ""

# --- Create virtual environment ---
if [ ! -d ".venv" ]; then
    echo "[1/3] Creating Python virtual environment..."
    python3 -m venv .venv
    echo "  -> .venv/ created"
else
    echo "[1/3] Virtual environment already exists (skipping)"
fi

# --- Activate and install dependencies ---
echo "[2/3] Installing dependencies..."
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "  -> Dependencies installed"

# --- Copy .env.example if .env doesn't exist ---
echo "[3/3] Configuring environment..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  -> .env created from .env.example"
    echo "  -> Edit .env with your API keys before running."
else
    echo "  -> .env already exists (skipping)"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys"
echo "  2. Activate the environment: source .venv/bin/activate"
echo "  3. Start the server:        uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo "  4. Test the health endpoint: curl http://localhost:8000/health"
