#!/bin/bash

cd "$(dirname -- "${BASH_SOURCE[0]}")/testray_automation_tasks"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 liferay/teams/headless/headless_testray.py