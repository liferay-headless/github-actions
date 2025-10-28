#!/usr/bin/env python3

import sys
import os
from typing import List
from jira import JIRA

# Ensure repo root is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from utils.liferay_utils.jira_utils.jira_liferay import get_jira_connection

# ✅ Paste your ordered list of open issues here
OPEN_ISSUES = [
    "LPD-62257",
    "LPD-68367",
    "LPD-64852",
    "LPD-55856",
    "LPD-55093",
    "LPD-59489",
    "LPD-46544",
    "LPD-68360",
    "LPD-55854",
    "LPD-55093",
    "LPD-55857",
    "LPD-46544",
    "LPD-66919",
    "LPD-40519",
    "LPD-64853",
    "LPD-66920",
    "LPD-48459",
    "LPD-61341",
    "LPD-64848",
    "LPD-64104",
    "LPD-59997",
    "LPD-67374",
    "LPD-66936",
    "LPD-61083",
    "LPD-66115",
    "LPD-66921",
    "LPD-67373",
]


def get_issue_details(jira_connection: JIRA, issue_keys: List[str]):
    """
    Fetch summary and components for each Jira issue key, preserving order.
    """
    results = []

    for issue_key in issue_keys:
        try:
            issue = jira_connection.issue(issue_key)

            summary = issue.fields.summary
            components = [c.name for c in issue.fields.components] or ["(No components)"]
            link = f"https://liferay.atlassian.net/browse/{issue_key}"

            results.append({
                "link": link,
                "summary": summary,
                "components": ", ".join(components),
            })

        except Exception as e:
            print(f"⚠️ Error retrieving {issue_key}: {e}")

    return results


# ------------------------ ENTRY POINT ------------------------
if __name__ == "__main__":
    jira_conn = get_jira_connection()
    details = get_issue_details(jira_conn, OPEN_ISSUES)
    jira_conn.close()

    # Print as a clean, aligned table
    print("\n--- Open Jira Issues (Detailed) ---\n")
    print(f"{'Link'} {'Summary'} {'Components'}")
    print("-" * 140)

    for item in details:
        print(f"{item['link']}, {item['summary']}, {item['components']}")