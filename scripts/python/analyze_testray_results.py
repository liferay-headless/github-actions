#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "jira==3.10.5",
#     "requests==2.32.5",
#     "sentence-transformers==5.1.0",
# ]
# ///

from utils.testray_helpers import (
    analyze_testflow,
    report_aft_ratio_for_latest,
)
from utils.testray_api import get_routine_to_builds

if __name__ == "__main__":
    builds = get_routine_to_builds()
    analyze_testflow(builds)
    report_aft_ratio_for_latest(builds)
