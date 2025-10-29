#!/bin/bash

cd "$(dirname -- "${BASH_SOURCE[0]}")/python"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 analyze_testray_results.py