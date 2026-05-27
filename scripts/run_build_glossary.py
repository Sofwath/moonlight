#!/usr/bin/env python3
"""Launch build-glossary with credentials from .env."""
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]

# Load .env
env_file = root / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

sys.path.insert(0, str(root))
sys.argv = ["moonlight", "build-glossary", "--model", "gemini-flash", "--budget", "10"]
from moonlight.cli import cli
cli()
