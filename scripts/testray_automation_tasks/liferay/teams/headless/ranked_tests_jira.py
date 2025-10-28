#!/usr/bin/env python3

import sys
import os
from typing import List, Tuple
from jira import JIRA

# Ensure repo root is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from utils.liferay_utils.jira_utils.jira_liferay import get_jira_connection

def find_lines_with_open_issues(jira_connection: JIRA, jira_lines: List[str]) -> List[Tuple[str, List[str]]]:
    """
    For each line containing one or more Jira IDs, keep the line if ANY issue in it is OPEN.
    Return tuples: (original line, list of open issue keys).
    """
    results = []

    for line in jira_lines:
        issue_keys = [key.strip() for key in line.split(",") if key.strip()]
        if not issue_keys:
            continue

        open_issues = []

        for issue_key in issue_keys:
            try:
                issue = jira_connection.issue(issue_key)
                status_name = issue.fields.status.name.strip().lower()

                if status_name == "open":
                    open_issues.append(issue_key)

            except Exception as e:
                print(f"⚠️ Error retrieving {issue_key}: {e}")

        if open_issues:
            results.append((line, open_issues))

    return results


# ------------------------ ENTRY POINT ------------------------
if __name__ == "__main__":
    jira_issues = [
        "LPD-62257",
        "LPD-46641, LPD-68367",
        "LPD-64852",
        "LPD-55856, LPD-66067, LPD-66918, LPD-67357",
        "LPD-55093",
        "LPD-59489",
        "LPD-46544",
        "LPD-65742, LPD-66076, LPD-66915, LPD-68360",
        "LPD-55854",
        "LPD-55093",
        "LPD-55857",
        "LPD-46544",
        "LPD-55855, LPD-66067, LPD-66919",
        "LPD-40519",
        "LPD-41106, LPD-64853",
        "LPD-66920",
        "LPD-48459",
        "LPD-61341, LPD-65739, LPD-67045",
        "LPD-64848, LPD-66069",
        "LPD-64104, LPD-64851, LPD-67227, LPD-67500",
        "LPD-59997",
        "LPD-66935, LPD-67374",
        "LPD-66936, LPD-67224",
        "LPD-61083, LPD-65391, LPD-67371",
        "LPD-66115, LPD-67226",
        "LPD-66921",
        "LPD-67373",
    ]

    jira_conn = get_jira_connection()
    results = find_lines_with_open_issues(jira_conn, jira_issues)
    jira_conn.close()

    if not results:
        print("✅ No open Jira issues found.")
    else:
        print("\n--- Open Jira Issues ---\n")
        for _, open_ids in results:
            print(", ".join(open_ids))
