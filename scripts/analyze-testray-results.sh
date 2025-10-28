#!/bin/bash

cd "$(dirname -- "${BASH_SOURCE[0]}")/testray_automation_tasks"
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd liferay/teams/headless
python3 headless_testray.py