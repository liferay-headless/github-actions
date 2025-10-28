#!/usr/bin/env python3

import json
from collections import defaultdict
from datetime import datetime, time  # FIX: datetime was used but not imported
from sentence_transformers import SentenceTransformer, util

from liferay.teams.headless.headless_contstants import ComponentMapping
from utils.liferay_utils.jira_utils.jira_helpers import (
    get_issue_status_by_key,
    create_jira_task,
    get_all_issues,
    close_issue,
)
from utils.liferay_utils.testray_utils.testray_api import *
from utils.liferay_utils.utilities import *

# Heavy model loaded once here (not in entrypoint)
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------------------------------------------------------
# Entry-point orchestration helpers
# ---------------------------------------------------------------------------

def get_latest_done_build(builds):
    """Return the newest build only if its import status is DONE; else None."""
    if not builds:
        return None
    latest_build = builds[0]
    if latest_build.get("importStatus", {}).get("key") != "DONE":
        print(f"✘ Latest build '{latest_build.get('name')}' is not DONE.")
        return None
    return latest_build


def prepare_task(jira_connection, builds, latest_build):
    """
    Ensure a task exists for latest_build and is actionable.
    Returns (task_id or None, latest_build_id).
    """
    latest_build_id = latest_build["id"]
    build_to_tasks = get_build_tasks(latest_build_id)

    if not build_to_tasks:
        print(f"[CREATE] No tasks for build '{latest_build['name']}', creating task and testflow.")
        task = create_task(latest_build)
        create_testflow(task["id"])
        print(f"✔ Using build {latest_build_id} and task {task['id']}")
        return task["id"], latest_build_id

    for task in build_to_tasks:
        due_status_key = task.get("dueStatus", {}).get("key")
        if due_status_key == "ABANDONED":
            print(f"Task {task['id']} has been ABANDONED.")
            return None, latest_build_id

        print(f"[USE] Using existing task {task['id']} with status {due_status_key}.")
        task_id = task["id"]

        status = get_task_status(task_id)
        if status.get("dueStatus", {}).get("key") == "COMPLETE":
            print(f"✔ Task {task_id} for build {latest_build_id} is now complete. No further processing required.")
            return None, latest_build_id

        print(f"✔ Using build {latest_build_id} and task {task_id}")
        return task_id, latest_build_id

    return None, latest_build_id


def _headless_epic_jql():
    _, quarter_number, year = get_current_quarter_info()
    return (
        f"text ~ '{year} Milestone {quarter_number} \\\\| Testing activities \\\\[Headless\\\\]' "
        f"and type = Epic and project='PUBLIC - Liferay Product Delivery' and status != Closed"
    )


def find_testing_epic(jira_connection):
    jql = _headless_epic_jql()
    related_epics = get_all_issues(jira_connection, jql, fields=["summary", "key"])
    print(f"✔ Retrieved {len(related_epics)} related Epics from JIRA")

    epic = related_epics[0] if len(related_epics) == 1 else None
    if epic:
        print(f"✔ Found testing epic: {epic}")
    else:
        print(f"✘ Expected 1 related epic, but found {len(related_epics)}")
    return epic


def maybe_autofill_from_previous(builds, latest_build):
    """
    If some previous build has a COMPLETED task, autofill into the latest build.
    """
    def _first_completed_build():
        for b in builds:
            for t in get_build_tasks(b["id"]):
                if t.get("dueStatus", {}).get("key") == "COMPLETE":
                    return b
        return None

    latest_complete = _first_completed_build()
    if latest_complete:
        print("Autofill from latest analysed build...")
        autofill_build(latest_complete["id"], latest_build["id"])
        print("✔ Completed")

# ---------------------------------------------------------------------------
# Subtask processing — scan → resolve (by error group) → stage → complete
# ---------------------------------------------------------------------------

def process_task_subtasks(*, task_id, latest_build_id, jira_connection, epic):
    """
    Iterate subtasks, detect unique failures grouped by error, reuse or create Jira tasks,
    and build batched updates and completion list.
    Returns (batch_updates, subtasks_to_complete, subtask_to_issues).
    """
    subtasks = get_task_subtasks(task_id)

    batch_updates = []
    subtasks_to_complete = []
    subtask_to_issues = defaultdict(set)

    for subtask in subtasks:
        subtask_id = subtask["id"]
        results = get_subtask_case_results(subtask_id)
        if not results:
            continue

        # Always collect any pre-existing result-level issues so they get bubbled up
        existing_issue_keys = _collect_result_issue_keys(results)
        if existing_issue_keys:
            subtask_to_issues[subtask_id].update(existing_issue_keys)

        # 1) Handle already-complete subtasks (backfill issues once if needed)
        if _is_subtask_complete(subtask):
            _backfill_subtask_issues_if_needed(subtask_id, subtask, results)
            continue

        # 2) Scan current results for unique failures (skip known errors)
        unique_failures, first_result_skipped = _scan_unique_failures(subtask_id, results)

        # Group failures by normalized error so each group can map to its own issue(s)
        groups = _group_failures_by_error(unique_failures)

        resolved_all_groups = True
        for error_key, group in groups.items():
            updates, issues_str, resolved = _resolve_unique_failures(
                jira_connection=jira_connection,
                epic=epic,
                latest_build_id=latest_build_id,
                task_id=task_id,
                subtask_id=subtask_id,
                unique_failures=group,
            )
            batch_updates.extend(updates)
            if issues_str:
                subtask_to_issues[subtask_id].add(issues_str)
            resolved_all_groups = resolved_all_groups and resolved

        # 3) Decide if subtask is fully handled
        no_unique_failures = len(unique_failures) == 0
        all_handled = first_result_skipped or no_unique_failures or resolved_all_groups

        # 4) Stage subtask for completion if everything is handled
        if all_handled:
            subtasks_to_complete.append(subtask_id)

    return batch_updates, subtasks_to_complete, subtask_to_issues


def finalize_task_completion(*, task_id, latest_build_id, jira_connection,
                             subtasks_to_complete, subtask_to_issues, batch_updates):
    """
    Apply batched updates, complete subtasks, close stale Jira issues, and complete the task.
    """
    # Apply batched case result updates first (assign issues to results)
    if batch_updates:
        assign_issue_to_case_result_batch(batch_updates)

    # Mark staged subtasks as COMPLETE (aggregating issues if provided)
    for subtask_id in subtasks_to_complete:
        issues_to_add = _join_issues(subtask_to_issues.get(subtask_id))
        print(f"✔ Marking subtask {subtask_id} as complete and associating issues: {issues_to_add}")
        update_subtask_status(subtask_id, issues=issues_to_add)

    # Check if all subtasks are done
    subtasks = get_task_subtasks(task_id)
    if not all(s.get("dueStatus", {}).get("key") == "COMPLETE" for s in subtasks):
        print(f"✔ Task {task_id} is not completed. Further processing required.")
        return

    # Close stale open routine tasks in Jira that were not reproduced in this run
    seen_issue_keys = _collect_issue_keys_from_subtasks(subtasks)
    _close_stale_routine_tasks(jira_connection, latest_build_id, seen_issue_keys)

    print(f"✔ All subtasks are complete, completing task {task_id}")
    complete_task(task_id)
    print(f"✔ Task {task_id} is now complete. No further processing required.")

# ---- scanning, grouping & resolving helpers -------------------------------------

def _is_subtask_complete(subtask):
    return subtask.get("dueStatus", {}).get("key") == "COMPLETE"


def _backfill_subtask_issues_if_needed(subtask_id, subtask, results):
    """
    When a subtask is COMPLETE but the aggregated 'issues' field is empty,
    aggregate from result-level 'issues' and write once.
    """
    if subtask.get("issues"):
        return
    issues = {r.get("issues") for r in results if r.get("issues")}
    if issues:
        issues_to_add = _join_issues(issues)
        update_subtask_status(subtask_id, issues=issues_to_add)


def _scan_unique_failures(subtask_id, results):
    """
    Return (unique_failures:list[dict], first_result_skipped:bool).
    We short-circuit the subtask if the first result matches a global skip.
    """
    unique_failures = []
    first_result = True
    first_result_skipped = False

    for r in results:
        error = (r.get("errors") or "")

        # First result can short-circuit the subtask
        if first_result and should_skip_result(error):
            update_subtask_status(subtask_id)
            first_result_skipped = True
            first_result = False
            continue

        first_result = False

        # Already handled or globally skippable
        if r.get("issues") or should_skip_result(error):
            continue

        # Consider as unique failure (unhandled)
        unique_failures.append({
            "error": error,
            "subtask_id": subtask_id,
            "case_id": r["r_caseToCaseResult_c_caseId"],
            "component_id": r.get("r_componentToCaseResult_c_componentId"),
            "result_id": r["id"]
        })

    return unique_failures, first_result_skipped


def _group_failures_by_error(unique_failures):
    """
    Group failures by normalized error so each group can map to its own Jira issue(s).
    """
    groups = defaultdict(list)
    for f in unique_failures:
        key = normalize_error(f["error"])
        groups[key].append(f)
    return groups


def _resolve_unique_failures(*, jira_connection, epic, latest_build_id, task_id, subtask_id, unique_failures):
    """
    Try to reuse similar open Jira issues; otherwise create an investigation.
    Returns (batch_updates, issues_str|None, resolved_bool).
    """
    if not unique_failures:
        return [], None, True

    # Reuse existing open issue(s) if the error is similar (lookup by the first item in this group)
    probe = unique_failures[0]
    has_similar_issue, blocked_dict = find_similar_open_issues(
        jira_connection,
        probe["case_id"],
        probe["error"],
    )

    if has_similar_issue and blocked_dict:
        issue_keys_str = blocked_dict["issues"]
        updates = [_blocked_update(f["result_id"], blocked_dict["dueStatus"], issue_keys_str) for f in unique_failures]
        return updates, issue_keys_str, True

    # Otherwise, create a brand-new investigation task for this group
    print(f"No similar issue found → create new investigation task for subtask {subtask_id}")
    issue = create_investigation_task_for_subtask(
        subtask_unique_failures=unique_failures,
        subtask_id=subtask_id,
        latest_build_id=latest_build_id,
        jira_connection=jira_connection,
        epic=epic,
        task_id=task_id,
        case_history_cache={},
    )

    if not issue:
        return [], None, False

    issue_key = issue.key
    updates = [_blocked_update(f["result_id"], {"key": "BLOCKED", "name": "Blocked"}, issue_key) for f in unique_failures]
    return updates, issue_key, True


def _blocked_update(result_id, due_status_dict, issues_str):
    return {"id": result_id, "dueStatus": due_status_dict, "issues": issues_str}


def _join_issues(issues_iterable):
    """
    Normalize a collection (or None) of issue strings into a single CSV or None.
    Each element may itself be a CSV; we split/trim/unique before joining.
    """
    if not issues_iterable:
        return None
    parts = set()
    for chunk in issues_iterable:
        if not chunk:
            continue
        for key in str(chunk).split(","):
            key = key.strip()
            if key:
                parts.add(key)
    if not parts:
        return None
    return ", ".join(sorted(parts))


def _collect_issue_keys_from_subtasks(subtasks):
    seen_issue_keys = set()
    for s in subtasks:
        issues_str = s.get("issues", "")
        if not issues_str:
            continue
        for k in issues_str.split(","):
            k = k.strip()
            if k:
                seen_issue_keys.add(k)
    return seen_issue_keys

def _collect_result_issue_keys(results):
    """
    From subtask results, collect any issue keys present in the `issues` field.
    Handles entries that may already be analyzed.
    """
    keys = set()
    for r in results:
        issues = r.get("issues")
        if not issues:
            continue
        for k in str(issues).split(","):
            k = k.strip()
            if k:
                keys.add(k)
    return keys

def _close_stale_routine_tasks(jira_connection, latest_build_id, seen_issue_keys):
    """
    Close open 'hl_routine_tasks' that did not appear in this run (not reproducible).
    """
    jql = "labels in ('hl_routine_tasks') AND labels not in ('test_fix') AND status = Open"
    open_jira_issues = get_all_issues(jira_connection, jql, fields=["key"])
    open_keys = {issue.key for issue in open_jira_issues}
    to_close = open_keys - seen_issue_keys
    if to_close:
        build_hash = get_current_build_hash(latest_build_id)
        print(f"ℹ Found {len(to_close)} issues to close as they are not reproducible in this run.")
        for issue_key in to_close:
            close_issue(jira_connection, issue_key, build_hash)

# ---------------------------------------------------------------------------
# KPI helper
# ---------------------------------------------------------------------------

def report_aft_ratio_for_latest(builds):
    """
    Compute and print AFT ratio KPI for latest DONE build vs beginning of quarter.
    (Same behavior as your previous get_automated_functional_tests_ratio flow, centralized here.)
    """
    latest_build = get_latest_done_build(builds)
    if not latest_build:
        return

    # Beginning-of-quarter build discovery
    quarter_start_date, _, _ = get_current_quarter_info()
    quarter_start = datetime.combine(quarter_start_date, time.min)

    best_build = None
    best_delta = None
    for b in builds:
        due_str = b.get("dueDate")
        if not due_str:
            continue
        dt = parse_execution_date(due_str)
        if not dt or dt < quarter_start:
            continue
        delta = dt - quarter_start
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_build = b

    if not best_build:
        print("✘ Could not find a build from the beginning of the quarter to calculate test ratio.")
        return

    latest_build_id = latest_build["id"]
    aft_case_type_id = get_case_type_id_by_name("Automated Functional Test")
    if not aft_case_type_id:
        print("✘ Could not find case type ID for 'Automated Functional Test'.")
        return

    print("⏳ Calculating automated functional test counts...")
    start_of_quarter_count = get_case_count_by_type_in_build(best_build["id"], aft_case_type_id)
    current_count = get_case_count_by_type_in_build(latest_build_id, aft_case_type_id)
    print("✔ Counts calculated.")

    report_poshi_tests_decrease(start_of_quarter_count, current_count)

# ---------------------------------------------------------------------------
# Existing domain logic (kept)
# ---------------------------------------------------------------------------

def handle_flaky_result(latest_build_id, jira_connection, result, case_id, result_error, epic):
    similar_open_issues = find_similar_open_issues(jira_connection, case_id, result_error, return_list=True)

    if similar_open_issues:
        print(f"Similar open issues: {similar_open_issues}")
        print(f"✔ Reassigning open issues {', '.join(similar_open_issues)} to result {result['id']}")
        return True, {
            "id": result["id"],
            "dueStatus": {"key": "TESTFIX", "name": "Test Fix"},
            "issues": ", ".join(similar_open_issues)
        }, None

    # Always create a new Test Fix task if none exist
    print("WE NEED TO CREATE SUBTASK: ")
    issue = create_testfix_task_for_subtask(
        case_id=case_id,
        latest_build_id=latest_build_id,
        jira_connection=jira_connection,
        epic=epic,
        result=result,
        result_error=result_error,
    )
    return True, {
        "id": result["id"],
        "dueStatus": {"key": "TESTFIX", "name": "Test Fix"},
        "issues": issue.key
    }, issue


def process_summary_result(
        summary_result,
        subtask_id,
        first_result,
        latest_build_id,
        jira_conn,
        epic,
        batch_updates,
        unique_tasks,
        case_ids,
        case_id_to_result,
        history_cache,
        task_id
):
    error = (summary_result.get("errors") or "")
    if should_skip_result(error):
        print(f"✔ Skipping result in subtask {subtask_id} due to known keyword")
        return True, None, False

    case_id = summary_result.get("r_caseToCaseResult_c_caseId")
    result_error = summary_result.get("errors", "")
    result_error_norm = normalize_error(result_error)

    case_id_to_result[case_id] = summary_result
    existing_issue = summary_result.get("issues")

    if first_result and should_skip_result(result_error):
        update_subtask_status(subtask_id)
        return True, None, False

    if existing_issue:
        return True, None, False

    if is_module_integration_test(case_id):
        is_flaky = False
    else:
        is_flaky = detect_flakiness(case_id, result_error_norm, history_cache)

    if is_flaky:
        handled, update_data, info = handle_flaky_result(
            latest_build_id, jira_conn, summary_result, case_id, result_error, epic
        )
        batch_updates.append(update_data)
        return True, None, False

    add_to_unique_tasks(unique_tasks, subtask_id, case_id, result_error)
    case_ids.add(case_id)
    return False, summary_result, True


def sort_cases_by_duration(subtask_case_pairs, case_duration_lookup):
    def safe_duration(c_id):
        d = case_duration_lookup.get(int(c_id))
        return d if isinstance(d, (int, float)) else float('inf')

    return sorted(subtask_case_pairs, key=lambda pair: safe_duration(pair[1]))


def build_case_rows(sorted_cases, case_duration_lookup, build_id, history_cache):
    printed_rows = []
    rca_info = None
    rca_batch = None
    rca_selector = None
    rca_compare = None

    component_name = "Unknown"

    for _, case_id, component_id in sorted_cases:
        try:
            case_info = get_case_info(case_id)
            case_name = case_info.get("name", "N/A")
            case_type_id = case_info.get("r_caseTypeToCases_c_caseTypeId")
            case_type_name = get_case_type_name(case_type_id) if case_type_id else "Unknown"
            component_name = get_component_name(component_id) if component_id else "Unknown"
            raw_duration = case_duration_lookup.get(int(case_id))
            duration = raw_duration if isinstance(raw_duration, (int, float)) else None

            passing_hash = get_last_passing_git_hash(case_id, build_id, history_cache)
            failing_hash = get_first_failing_git_hash(case_id, build_id, history_cache)

            github_compare = (
                f"https://github.com/liferay/liferay-portal/compare/{passing_hash}...{failing_hash}"
                if passing_hash and failing_hash else "###"
            )

            batch_name, test_selector = get_batch_info(case_name, case_type_name)

            if not rca_info and batch_name and test_selector:
                rca_info = build_rca_block(batch_name, test_selector, github_compare)
                rca_batch = batch_name
                rca_selector = test_selector
                rca_compare = github_compare

            elif not rca_info:
                rca_info = f"\nCompare: {github_compare}"

            row = [case_name, format_duration(duration), component_name]
            printed_rows.append(row)

        except Exception as e:
            print(f"[ERROR] Failed to fetch data for case_id={case_id} → {e}")

    return printed_rows, rca_info, rca_batch, rca_selector, rca_compare, component_name

def get_last_passing_git_hash(case_id, build_id, history_cache):
    entire_history = history_cache.get(case_id)
    if entire_history is None:
        entire_history = get_case_result_history_for_routine(case_id)
        history_cache[case_id] = entire_history

    result_history_for_build = filter_case_result_history_by_build(entire_history, build_id)
    if not result_history_for_build:
        return None

    failing_hash_execution_date = result_history_for_build[0].get('executionDate')
    item = get_last_passing_result(entire_history, failing_hash_execution_date)
    last_passing_hash = item.get('gitHash') if item else None
    return last_passing_hash


def get_first_failing_git_hash(case_id, build_id, history_cache):
    """
    Find the first failing git hash after the last passing run for this case.
    """
    entire_history = history_cache.get(case_id)
    if entire_history is None:
        entire_history = get_case_result_history_for_routine(case_id)
        history_cache[case_id] = entire_history

    if not entire_history:
        return None

    result_history_for_build = filter_case_result_history_by_build(entire_history, build_id)
    if not result_history_for_build:
        return None

    failing_execution_date = result_history_for_build[0].get("executionDate")

    last_passing = get_last_passing_result(entire_history, failing_execution_date)
    if not last_passing:
        return result_history_for_build[0].get("gitHash")

    last_pass_date = parse_execution_date(last_passing["executionDate"])
    for item in reversed(entire_history):
        exec_date = parse_execution_date(item.get("executionDate"))
        if not exec_date:
            continue
        if exec_date > last_pass_date and item.get("status") in STATUS_FAILED_BLOCKED_TESTFIX:
            return item.get("gitHash")

    return None

def get_batch_info(case_name, case_type_name):
    if case_type_name == "Playwright Test":
        selector = case_name.split(" >")[0] if " >" in case_name else case_name
        return "playwright-js-tomcat101-postgresql163", selector
    elif case_type_name == "Automated Functional Test":
        return "functional-tomcat101-postgresql163", case_name
    elif case_type_name == "Modules Integration Test":
        trimmed_name = case_name.split(".")[-1]
        return "modules-integration-postgresql163", f"\\*\\*/src/testIntegration/\\*\\*/{trimmed_name}.java"
    return None, None


def build_rca_block(batch_name, test_selector, github_compare):
    return (
        "\nParameters to run Root Cause Analysis on https://test-1-1.liferay.com/job/root-cause-analysis-tool/ :\n"
        f"PORTAL_BATCH_NAME: {batch_name}\n"
        f"PORTAL_BATCH_TEST_SELECTOR: {test_selector}\n"
        f"PORTAL_BRANCH_SHAS: {github_compare}\n"
        f"PORTAL_GITHUB_URL: https://github.com/liferay/liferay-portal/tree/master\n"
        f"PORTAL_UPSTREAM_BRANCH_NAME: master"
    )


def build_rca_html_block(batch_name, test_selector, github_compare):
    return (
            "<p>Parameters to run "
            "<a href='https://test-1-1.liferay.com/job/root-cause-analysis-tool/' target='_blank'>"
            "Root Cause Analysis Tool</a>:</p>"
            "<pre><b>PORTAL_BATCH_NAME</b>: " + batch_name + "\n"
                                                             "<b>PORTAL_BATCH_TEST_SELECTOR</b>: " + test_selector + "\n"
                                                                                                                     "<b>PORTAL_BRANCH_SHAS</b>: <a href='" + github_compare + "' target='_blank'>" + github_compare + "</a>\n"
                                                                                                                                                                                                                       "<b>PORTAL_GITHUB_URL</b>: <a href='https://github.com/liferay/liferay-portal/tree/master' target='_blank'>"
                                                                                                                                                                                                                       "https://github.com/liferay/liferay-portal/tree/master</a>\n"
                                                                                                                                                                                                                       "<b>PORTAL_UPSTREAM_BRANCH_NAME</b>: master</pre>"
    )


def get_build_from_beginning_of_current_quarter(builds):
    quarter_start_date, _, _ = get_current_quarter_info()
    quarter_start = datetime.combine(quarter_start_date, time.min)

    best_build = None
    best_delta = None

    for build in builds:
        due_str = build.get("dueDate")
        if not due_str:
            continue

        dt = parse_execution_date(due_str)
        if not dt or dt < quarter_start:
            continue

        delta = dt - quarter_start
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_build = build

    return best_build["id"] if best_build else None


def find_or_create_task(build, jira_connection, latest_build_id):
    build_to_tasks = get_build_tasks(build["id"])

    if not build_to_tasks:
        print(f"[CREATE] No tasks for build '{build['name']}', creating task and testflow.")
        task = create_task(build)
        create_testflow(task["id"])
        return task["id"]

    for task in build_to_tasks:
        due_status_key = task.get("dueStatus", {}).get("key")
        if due_status_key == "ABANDONED":
            print(f"Task {task['id']} has been ABANDONED.")
            return None
        print(f"[USE] Using existing task {task['id']} with status {due_status_key}.")
        task_id = task["id"]
        subtasks = get_task_subtasks(task_id)
        if check_and_complete_task_if_all_subtasks_done(task_id, subtasks, jira_connection, latest_build_id):
            return None
        return task_id
    return None


def get_latest_build_with_completed_task(builds):
    for build in builds:
        build_tasks = get_build_tasks(build["id"])
        for task in build_tasks:
            if task.get("dueStatus", {}).get("key") == "COMPLETE":
                return build
    return None


def is_handled(result):
    err = (result.get("errors") or "")
    return bool(result.get("issues")) or should_skip_result(err)


def should_skip_result(error):
    if "AssertionError" in error:
        return False

    skip_error_keywords = [
        "Failed prior to running test",
        "PortalLogAssertorTest#testScanXMLLog",
        "Skipped test",
        "The build failed prior to running the test",
        "test-portal-testsuite-upstream-downstream(master) timed out after",
        "TEST_SETUP_ERROR",
        "Unable to run test on CI"
    ]
    return any(keyword in (error or "") for keyword in skip_error_keywords)


def build_flaky_result_metadata(latest_build_id, result, case_id, result_error):
    case_info = get_case_info(case_id)
    return {
        "link": f"https://testray.liferay.com/web/testray#/project/35392/routines/{HEADLESS_ROUTINE_ID}/build/{latest_build_id}/case-result/{result['id']}",
        "test_name": case_info.get("name", "N/A"),
        "error": result_error,
        "component": get_component_name(case_info.get("r_componentToCases_c_componentId")) or "Unknown"
    }


def create_testfix_task_for_subtask(
        case_id,
        latest_build_id,
        jira_connection,
        epic,
        result,
        result_error
):
    """
    Creates a Test Fix task in Jira when a flaky result is detected and
    no similar open issues exist. It includes context about the failing case,
    component, and error for quicker triage.
    """
    case_info = get_case_info(case_id)
    component_name = get_component_name(case_info.get("r_componentToCases_c_componentId")) or "Unknown"

    # Build description
    description_lines = [
        "*Flaky Test Detected in Testray*",
        f"[Testray Result|https://testray.liferay.com/web/testray#/project/35392/routines/{HEADLESS_ROUTINE_ID}/build/{latest_build_id}/case-result/{result['id']}]",
        "",
        "h3. Error",
        f"{{code}}{result_error}{{code}}",
        "",
        "h3. Test Details",
        f"*Name:* {case_info.get('name', 'N/A')}",
        f"*Component:* {component_name}",
        f"*Result ID:* {result['id']}",
    ]

    summary = f"Test Fix: {case_info.get('name', 'N/A')} - {result_error[:80]}"
    description = "\n".join(description_lines)

    jira_components = [
        ComponentMapping.TestrayToJira.get(component_name, component_name)
    ]

    # Create Jira issue
    issue = create_jira_task(
        jira_local=jira_connection,
        epic=epic,
        summary=summary,
        description=description,
        component=jira_components,
        label="test_fix"
    )

    print(f"✔ Created Test Fix task for case {case_id}: {issue.key}")
    return issue


def find_similar_open_issues(jira_connection, case_id, result_error, *, return_list=False):
    """
    Look for similar errors in history that have open Jira issues.

    Returns:
        If return_list=True: List[str]
        Else: Tuple[bool, dict or None]
    """
    seen_issues = set()
    similar_open_issues = []

    history = get_case_result_history_for_routine_not_passed(case_id)
    result_error_norm = normalize_error(result_error)

    for past_result in history:
        issues_str = past_result.get("issues", "")
        if not issues_str:
            continue

        issue_keys = [key.strip() for key in issues_str.split(",")]
        open_issues = []

        for issue_key in issue_keys:
            if issue_key in seen_issues:
                continue
            try:
                _, status = get_issue_status_by_key(jira_connection, issue_key)
                if status != "Closed":
                    open_issues.append(issue_key)
                seen_issues.add(issue_key)
            except Exception as e:
                print(f"Error retrieving issue {issue_key}: {e}")

        if not open_issues:
            continue

        history_error = past_result.get("error", "")
        history_error_norm = normalize_error(history_error)

        if are_errors_similar(result_error_norm, history_error_norm):
            if return_list:
                similar_open_issues.extend(open_issues)
                return similar_open_issues  # stop at first
            else:
                return True, {
                    "dueStatus": {"key": "BLOCKED", "name": "Blocked"},
                    "issues": ", ".join(open_issues)
                }

    return similar_open_issues if return_list else (False, None)


def handle_module_integration_flaky(latest_build_id, result, case_id, result_error, similar_open_issues):
    print(f"📌 Case ID {case_id} is a Modules Integration Test. Evaluate manually if it is flaky.")
    error_link = get_error_messages_link(result)
    if error_link:
        print(f"🔗 Error Messages Link: {error_link}")
    if similar_open_issues:
        print(f"🛠 Similar open LPD issues found: {', '.join(similar_open_issues)}")

    return False, None, build_flaky_result_metadata(latest_build_id, result, case_id, result_error)


def add_to_unique_tasks(unique_tasks, subtask_id, case_id, error):
    unique_tasks.append({
        "subtask_id": subtask_id,
        "case_id": case_id,
        "error": error
    })


def detect_flakiness(case_id, current_error_norm, history_cache):
    history = history_cache.get(case_id)
    if history is None:
        history = get_case_result_history_for_routine(case_id)
        history_cache[case_id] = history

    if not history:
        return False, "no_history"

    fail_count = 0
    pass_count = 0
    switch_count = 0
    last_status = None
    similar_error_failures = 0
    unique_errors = set()

    for idx, result in enumerate(history):
        status = result.get("status")
        raw_error = result.get("error", "")
        error = normalize_error(raw_error)

        if status in STATUS_FAILED_BLOCKED_TESTFIX:
            fail_count += 1
            unique_errors.add(error)
            is_similar = are_errors_similar(current_error_norm, error)
            if is_similar:
                similar_error_failures += 1
        elif status == "PASSED":
            pass_count += 1

        if last_status and last_status != status:
            switch_count += 1
        last_status = status

    total = pass_count + fail_count

    if total < 5:
        return False, "insufficient_data"

    flakiness_score = switch_count / (total - 1)
    failure_rate = fail_count / total
    similar_error_ratio = similar_error_failures / fail_count

    is_flaky = (
            flakiness_score > 0.12 and
            0.05 < failure_rate < 0.75 and
            similar_error_ratio >= 0.9
    )

    return is_flaky


def report_poshi_tests_decrease(start_of_quarter_count, current_count):
    if start_of_quarter_count == 0:
        print("Cannot calculate decrease percentage (division by zero).")
        return

    items_less = start_of_quarter_count - current_count
    decrease_percent = (items_less / start_of_quarter_count) * 100

    if decrease_percent < 10.0:
        print(
            f"The total number of POSHI tests has gone down by {decrease_percent:.2f}% "
            f"compared to what it was at the beginning of the quarter. "
            f"We're targeting a 10% decrease, so there's still work to do."
        )
    else:
        print(
            f"The total number of POSHI tests has gone down by {decrease_percent:.2f}% "
            f"compared to what it was at the beginning of the quarter. "
            f"KPI of 10% accomplished, but keep pushing!"
        )


def is_automated_functional_test(c_id):
    case_info = get_case_info(c_id)
    case_type_id = case_info.get("r_caseTypeToCases_c_caseTypeId")
    return get_case_type_name(case_type_id) == "Automated Functional Test"


def is_module_integration_test(c_id):
    case_info = get_case_info(c_id)
    case_type_id = case_info.get("r_caseTypeToCases_c_caseTypeId")
    return get_case_type_name(case_type_id) == "Modules Integration Test"


def check_and_complete_task_if_all_subtasks_done(task_id, subtasks, jira_connection, latest_build_id):
    all_complete = all(subtask.get("dueStatus", {}).get("key") == "COMPLETE" for subtask in subtasks)

    if all_complete:
        # Collect all unique issue keys from the completed subtasks
        testray_issue_keys = set()
        for subtask in subtasks:
            issues_str = subtask.get("issues", "")
            if issues_str:
                testray_issue_keys.update(key.strip() for key in issues_str.split(','))

        # Fetch open routine tasks from Jira
        jql = "labels in ('hl_routine_tasks') AND labels not in ('test_fix') AND status = Open"
        open_jira_issues = get_all_issues(jira_connection, jql, fields=["key"])
        open_jira_issue_keys = {issue.key for issue in open_jira_issues}

        # Find issues that are open in Jira but not present in our completed TestRay task
        issues_to_close = open_jira_issue_keys - testray_issue_keys

        if issues_to_close:
            build_hash = get_current_build_hash(latest_build_id)
            print(f"ℹ Found {len(issues_to_close)} issues to close as they are not reproducible in this run.")
            for issue_key in issues_to_close:
                close_issue(jira_connection, issue_key, build_hash)

        print(f"✔ All subtasks are complete, completing task {task_id}")
        complete_task(task_id)
        return True  # Task completed
    else:
        print(f"ℹ Not all subtasks are complete for task {task_id}, task remains in progress")
        return False  # Task still open


def are_errors_similar(current_norm, history_norm, threshold=0.8):
    """
    Compare two error messages semantically using sentence embeddings.
    """
    emb_a = model.encode(current_norm, convert_to_tensor=True)
    emb_b = model.encode(history_norm, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb_a, emb_b).item()
    return similarity >= threshold


def group_errors_by_type(unique_tasks):
    error_to_cases = defaultdict(list)
    for item in unique_tasks:
        error_to_cases[item["error"]].append((item["subtask_id"], item["case_id"], item["component_id"]))
    return error_to_cases


def build_case_duration_lookup(unique_tasks, build_id):
    raw_build_results = get_all_build_case_results(build_id)
    interested_case_ids = {
        int(item["case_id"]) for item in unique_tasks if item.get("case_id")
    }

    return {
        int(item["r_caseToCaseResult_c_caseId"]): item.get("duration")
        for item in raw_build_results
        if item.get("r_caseToCaseResult_c_caseId")
           and int(item["r_caseToCaseResult_c_caseId"]) in interested_case_ids
    }


def get_error_messages_link(result_summary):
    case_result_id = result_summary.get("id")
    if not case_result_id:
        print("Warning: No 'id' found in result_summary")

    failure_url = fetch_failure_url(case_result_id)
    if not failure_url:
        print("No failure message URL found in attachments.")
    return failure_url


def is_flagged_as_flaky(case_id):
    info = get_case_info(case_id)
    flaky_flag = info.get('flaky', None)

    if flaky_flag is None:
        print(f"[WARNING] Flaky flag doesn't exist for case_id={case_id}")

    return flaky_flag or False


def fetch_failure_url(case_result_id):
    detailed_result = get_case_result(case_result_id)
    attachments_json_str = detailed_result.get('attachments', '[]')

    try:
        attachments = json.loads(attachments_json_str)
        for attachment in attachments:
            if attachment.get('name') == "Failure Messages":
                return attachment.get('url')
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse attachments JSON: {e}")

    return None


def get_case_result_history_for_routine(case_id):
    items = fetch_case_results(case_id, HEADLESS_ROUTINE_ID)
    return sort_by_execution_date_desc(items)


def get_case_result_history_for_routine_not_passed(case_id):
    items = fetch_case_results(case_id, HEADLESS_ROUTINE_ID, status=STATUS_FAILED_BLOCKED_TESTFIX)
    return sort_by_execution_date_desc(items)


def sort_by_execution_date_desc(items):
    def get_sort_key(item):
        date_str = item.get("executionDate", "")
        parsed_date = parse_execution_date(date_str)
        return parsed_date or datetime.min

    return sorted(items, key=get_sort_key, reverse=True)


def get_last_passing_result(entire_history, max_execution_date):
    status_passed = "PASSED"

    if isinstance(max_execution_date, str):
        max_execution_date = parse_execution_date(max_execution_date)
        if not max_execution_date:
            print("❌ Invalid max_due_date format")
            return None

    last_passing = None
    last_date = None

    for item in entire_history:
        if item.get("status") != status_passed:
            continue

        execution_date_str = item.get("executionDate")
        if not execution_date_str:
            continue

        execution_date = parse_execution_date(execution_date_str)
        if not execution_date or execution_date >= max_execution_date:
            continue

        if last_date is None or execution_date > last_date:
            last_passing = item
            last_date = execution_date

    return last_passing


def filter_case_result_history_by_build(history, build_id):
    """Filter case result history by build ID."""
    return [
        item for item in history
        if item.get("testrayBuildId") == build_id
    ]

def get_current_build_hash(build_id):
    build = get_build_info(build_id)
    git_hash = build.get('gitHash')
    return git_hash


def get_task_routine_id(task_id):
    build_id = get_task_build_id(task_id)
    build_info = get_build_info(build_id)
    return build_info["r_routineToBuilds_c_routineId"]


def create_investigation_task_for_subtask(
        subtask_unique_failures,
        subtask_id,
        latest_build_id,
        jira_connection,
        epic,
        task_id,
        case_history_cache
):
    """
    Creates an investigation task in Jira for a subtask with unique failures.
    Groups failures by error, outputs a Jira-friendly description with a table of
    test names, components, duration, and RCA details (once).
    """
    # Group by error
    error_to_cases = group_errors_by_type(subtask_unique_failures)
    case_duration_lookup = build_case_duration_lookup(subtask_unique_failures, latest_build_id)

    description_lines = [
        "*Unique Failures in Testray Subtask*",
        f"[Testray Subtask|https://testray.liferay.com/web/testray#/testflow/{task_id}/subtasks/{subtask_id}]",
        "",
    ]

    first_error = None
    rca_included = False
    component_name = None

    for error, subtask_case_pairs in error_to_cases.items():
        if not first_error:
            first_error = error[:80]  # Jira summary

        description_lines.append("h3. Error")
        description_lines.append(f"{{code}}{error}{{code}}")

        sorted_cases = sort_cases_by_duration(subtask_case_pairs, case_duration_lookup)
        printed_rows, rca_info, batch_name, test_selector, github_compare, component_name = build_case_rows(
            sorted_cases, case_duration_lookup, latest_build_id, case_history_cache
        )

        description_lines.append("")
        description_lines.append("|| Test Name || Component || Duration ||")
        for row in printed_rows:
            name, duration, component = row
            description_lines.append(f"| {name} | {component} | {duration} |")

        if not rca_included and batch_name and test_selector and github_compare:
            description_lines.append("")
            description_lines.append("h3. RCA Details")
            description_lines.append("")
            description_lines.append(f"*Batch:* {batch_name}")
            description_lines.append(f"*Test Selector:* {test_selector}")
            description_lines.append(f"*GitHub Compare:* {github_compare}")
            rca_included = True

    summary = f"Investigate {first_error}..."
    description = "\n".join(description_lines)
    jira_components = [
        ComponentMapping.TestrayToJira.get(c, c)  # fallback to original if not mapped
        for c in (component_name or "Unknown").split(",")
    ]

    # Create Jira issue
    issue = create_jira_task(
        jira_local=jira_connection,
        epic=epic,
        summary=summary,
        description=description,
        component=jira_components,
        label=None
    )

    print(f"✔ Created investigation task for subtask {subtask_id}: {issue.key}")
    return issue