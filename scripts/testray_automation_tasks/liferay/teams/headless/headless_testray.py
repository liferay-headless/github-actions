#!/usr/bin/env python3

import os
import sys

# Keep your path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from utils.liferay_utils.testray_utils.testray_helpers import (
    get_latest_done_build,
    prepare_task,
    find_testing_epic,
    maybe_autofill_from_previous,
    process_task_subtasks,
    finalize_task_completion,
    report_aft_ratio_for_latest,
)
from utils.liferay_utils.jira_utils.jira_liferay import get_jira_connection
from utils.liferay_utils.testray_utils.testray_api import get_routine_to_builds


def analyze_testflow(jira_connection, builds):
    """
    Slim orchestration:
      1) find latest DONE build + ensure task exists
      2) fetch testing epic + maybe autofill from previous completed task
      3) process subtasks & results (collect updates only)
      4) apply updates and attempt task completion/cleanup
    """
    latest_build = get_latest_done_build(builds)
    if not latest_build:
        return

    task_id, latest_build_id = prepare_task(jira_connection, builds, latest_build)
    if not task_id:
        print("✘ Could not find or create a valid task, exiting.")
        return

    epic = find_testing_epic(jira_connection)
    maybe_autofill_from_previous(builds, latest_build)

    batch_updates, subtasks_to_complete, subtask_to_issues = process_task_subtasks(
        task_id=task_id,
        latest_build_id=latest_build_id,
        jira_connection=jira_connection,
        epic=epic,
    )

    finalize_task_completion(
        task_id=task_id,
        latest_build_id=latest_build_id,
        jira_connection=jira_connection,
        subtasks_to_complete=subtasks_to_complete,
        subtask_to_issues=subtask_to_issues,
        batch_updates=batch_updates,
    )


def main():
    jira_conn = get_jira_connection()
    builds = get_routine_to_builds()
    analyze_testflow(jira_conn, builds)
    report_aft_ratio_for_latest(builds)


if __name__ == "__main__":
    main()