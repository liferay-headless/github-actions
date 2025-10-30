import re
from collections import defaultdict
from datetime import datetime, time
from sentence_transformers import SentenceTransformer, util
from functools import lru_cache
from utils.jira_helpers import (
    get_jira_connection,
    get_issue_status_by_key,
    create_jira_task,
    get_all_issues,
    close_issue,
)
from utils.testray_api import (
    STATUS_FAILED_BLOCKED_TESTFIX,
    HEADLESS_ROUTINE_ID,
    get_build_tasks,
    create_testflow,
    get_build_info,
    fetch_case_results,
    get_all_build_case_results,
    get_component_name,
    create_task,
    get_task_status,
    autofill_build,
    get_task_subtasks,
    get_subtask_case_results,
    assign_issue_to_case_result_batch,
    update_subtask_status,
    complete_task,
    get_case_type_id_by_name,
    get_case_count_by_type_in_build,
    get_case_info,
    get_case_type_name,
)


class ComponentMapping:
    TestrayToJira = {
        "API Builder": "API Builder",
        "Connectors": "Data Integration > Connectors",
        "Data Migration Center": "Data Integration > Data Migration Center",
        "Export/Import": "Data Integration > Export/Import",
        "Headless Batch Engine API": "Headless Batch Engine API",
        "Headless Discovery Application": "Headless Discovery Application",
        "Job Scheduler": "Data Integration > Job Scheduler",
        "Object": "Objects > Object Entries REST APIs",
        "Object Entries REST APIs": "Objects > Object Entries REST APIs",
        "REST Builder": "REST Builder",
        "REST Infrastructure": "REST Infrastructure",
        "Site Templates": "Content Publishing > Site Templates",
        "Staging": "Data Integration > Staging",
        "Upgrades Staging": "Data Integration > Staging",  # Assuming same as 'Staging'
    }


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
        print(
            f"[CREATE] No tasks for build '{latest_build['name']}', creating task and testflow."
        )
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
            print(
                f"✔ Task {task_id} for build {latest_build_id} is now complete. No further processing required."
            )
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


def normalize_error(error):
    """Normalize and clean error messages for comparison and pattern matching."""
    if not error:
        return ""

    # Collapse whitespace
    error = " ".join(error.strip().split())

    # Remove timestamps, memory addresses, test durations, or dynamic values
    error = re.sub(r"\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}", "", error)  # timestamps
    error = re.sub(r"0x[0-9A-Fa-f]+", "", error)  # memory addresses
    error = re.sub(r"\d+\s*(ms|s|seconds|minutes|m)", "", error)  # durations
    error = re.sub(r'".*?"', '"..."', error)  # replace quoted strings with placeholder

    return error


def parse_execution_date(date_str):
    date_str = date_str.strip().rstrip("Z").replace("T", " ")

    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def find_testing_epic(jira_connection):
    jql = _headless_epic_jql()
    related_epics = get_all_issues(jira_connection, jql, fields=["summary", "key"])
    print(f"✔ Retrieved {len(related_epics)} related Epics from JIRA")

    epic = related_epics[0] if len(related_epics) == 1 else None
    print(
        f"✔ Found testing epic: {epic}"
        if epic
        else f"✘ Expected 1 related epic, but found {len(related_epics)}"
    )
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
        unique_failures, first_result_skipped = _scan_unique_failures(
            subtask_id, results
        )

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


def finalize_task_completion(
    *,
    task_id,
    latest_build_id,
    jira_connection,
    subtasks_to_complete,
    subtask_to_issues,
    batch_updates,
):
    """
    Apply batched updates, complete subtasks, close stale Jira issues, and complete the task.
    """
    # Apply batched case result updates first (assign issues to results)
    if batch_updates:
        assign_issue_to_case_result_batch(batch_updates)

    # Mark staged subtasks as COMPLETE (aggregating issues if provided)
    for subtask_id in subtasks_to_complete:
        issues_to_add = _join_issues(subtask_to_issues.get(subtask_id))
        print(
            f"✔ Marking subtask {subtask_id} as complete and associating issues: {issues_to_add}"
        )
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
        update_subtask_status(subtask_id, issues=_join_issues(issues))


def _scan_unique_failures(subtask_id, results):
    """
    Return (unique_failures:list[dict], first_result_skipped:bool).
    We short-circuit the subtask if the first result matches a global skip.
    """
    unique_failures = []
    first_result = True
    first_result_skipped = False

    for r in results:
        error = r.get("errors") or ""

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
        unique_failures.append(
            {
                "error": error,
                "subtask_id": subtask_id,
                "case_id": r["r_caseToCaseResult_c_caseId"],
                "component_id": r.get("r_componentToCaseResult_c_componentId"),
                "result_id": r["id"],
            }
        )

    return unique_failures, first_result_skipped


def _group_failures_by_error(unique_failures):
    """
    Group failures by normalized error so each group can map to its own Jira issue(s).
    """
    groups = defaultdict(list)
    for f in unique_failures:
        groups[normalize_error(f["error"])].append(f)
    return groups


def _resolve_unique_failures(
    *, jira_connection, epic, latest_build_id, task_id, subtask_id, unique_failures
):
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
        updates = [
            _blocked_update(f["result_id"], blocked_dict["dueStatus"], issue_keys_str)
            for f in unique_failures
        ]
        return updates, issue_keys_str, True

    # Otherwise, create a brand-new investigation task for this group
    print(
        f"No similar issue found → create new investigation task for subtask {subtask_id}"
    )
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
    updates = [
        _blocked_update(
            f["result_id"], {"key": "BLOCKED", "name": "Blocked"}, issue_key
        )
        for f in unique_failures
    ]
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
    parts = {
        key.strip()
        for chunk in issues_iterable
        if chunk
        for key in str(chunk).split(",")
        if key.strip()
    }
    return ", ".join(sorted(parts)) if parts else None


def _collect_issue_keys_from_subtasks(subtasks):
    return {
        k.strip()
        for s in subtasks
        for k in str(s.get("issues", "")).split(",")
        if k.strip()
    }


def _collect_result_issue_keys(results):
    """
    From subtask results, collect any issue keys present in the `issues` field.
    Handles entries that may already be analyzed.
    """
    return {
        k.strip()
        for r in results
        if r.get("issues")
        for k in str(r["issues"]).split(",")
        if k.strip()
    }


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
        print(
            f"ℹ Found {len(to_close)} issues to close as they are not reproducible in this run."
        )
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
        print(
            "✘ Could not find a build from the beginning of the quarter to calculate test ratio."
        )
        return

    latest_build_id = latest_build["id"]
    aft_case_type_id = get_case_type_id_by_name("Automated Functional Test")
    if not aft_case_type_id:
        print("✘ Could not find case type ID for 'Automated Functional Test'.")
        return

    print("⏳ Calculating automated functional test counts...")
    start_of_quarter_count = get_case_count_by_type_in_build(
        best_build["id"], aft_case_type_id
    )
    current_count = get_case_count_by_type_in_build(latest_build_id, aft_case_type_id)
    print("✔ Counts calculated.")

    report_poshi_tests_decrease(start_of_quarter_count, current_count)


# ---------------------------------------------------------------------------
# Existing domain logic (kept)
# ---------------------------------------------------------------------------


def sort_cases_by_duration(subtask_case_pairs, case_duration_lookup):
    def safe_duration(c_id):
        d = case_duration_lookup.get(int(c_id))
        return d if isinstance(d, (int, float)) else float("inf")

    return sorted(subtask_case_pairs, key=lambda pair: safe_duration(pair[1]))


def format_duration(ms):
    """Convert milliseconds into human-readable duration."""
    if not isinstance(ms, (int, float)):
        return "N/A"
    minutes = int(ms // 60000)
    seconds = int((ms % 60000) // 1000)
    return f"{minutes}m {seconds}s"


def get_current_quarter_info():
    """
    Returns:
        quarter_start (datetime.date): Start date of the current quarter.
        quarter_number (int): Quarter number (1, 2, 3, or 4).
        year (int): Current year.
    """
    today = datetime.today()
    quarter_number = (today.month - 1) // 3 + 1
    start_month = (quarter_number - 1) * 3 + 1
    quarter_start = datetime(today.year, start_month, 1).date()
    return quarter_start, quarter_number, today.year


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
            case_type_name = (
                get_case_type_name(case_type_id) if case_type_id else "Unknown"
            )
            component_name = (
                get_component_name(component_id) if component_id else "Unknown"
            )
            raw_duration = case_duration_lookup.get(int(case_id))
            duration = raw_duration if isinstance(raw_duration, (int, float)) else None

            passing_hash = get_last_passing_git_hash(case_id, build_id, history_cache)
            failing_hash = get_first_failing_git_hash(case_id, build_id, history_cache)

            github_compare = (
                f"https://github.com/liferay/liferay-portal/compare/{passing_hash}...{failing_hash}"
                if passing_hash and failing_hash
                else "###"
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

    result_history_for_build = filter_case_result_history_by_build(
        entire_history, build_id
    )
    if not result_history_for_build:
        return None

    failing_hash_execution_date = result_history_for_build[0].get("executionDate")
    item = get_last_passing_result(entire_history, failing_hash_execution_date)
    last_passing_hash = item.get("gitHash") if item else None
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

    result_history_for_build = filter_case_result_history_by_build(
        entire_history, build_id
    )
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
        if (
            exec_date > last_pass_date
            and item.get("status") in STATUS_FAILED_BLOCKED_TESTFIX
        ):
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
        return (
            "modules-integration-postgresql163",
            f"\\*\\*/src/testIntegration/\\*\\*/{trimmed_name}.java",
        )
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
        "Unable to run test on CI",
    ]
    return any(keyword in (error or "") for keyword in skip_error_keywords)


def find_similar_open_issues(
    jira_connection, case_id, result_error, *, return_list=False
):
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
                    "issues": ", ".join(open_issues),
                }

    return similar_open_issues if return_list else (False, None)


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


@lru_cache()
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def are_errors_similar(current_norm, history_norm, threshold=0.8):
    """
    Compare two error messages semantically using sentence embeddings.
    """
    model = load_model()
    emb_a = model.encode(current_norm, convert_to_tensor=True)
    emb_b = model.encode(history_norm, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb_a, emb_b).item()
    return similarity >= threshold


def group_errors_by_type(unique_tasks):
    error_to_cases = defaultdict(list)
    for item in unique_tasks:
        error_to_cases[item["error"]].append(
            (item["subtask_id"], item["case_id"], item["component_id"])
        )
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


def get_case_result_history_for_routine(case_id):
    items = fetch_case_results(case_id, HEADLESS_ROUTINE_ID)
    return sort_by_execution_date_desc(items)


def get_case_result_history_for_routine_not_passed(case_id):
    items = fetch_case_results(
        case_id, HEADLESS_ROUTINE_ID, status=STATUS_FAILED_BLOCKED_TESTFIX
    )
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
    return [item for item in history if item.get("testrayBuildId") == build_id]


def get_current_build_hash(build_id):
    build = get_build_info(build_id)
    git_hash = build.get("gitHash")
    return git_hash


def create_investigation_task_for_subtask(
    subtask_unique_failures,
    subtask_id,
    latest_build_id,
    jira_connection,
    epic,
    task_id,
    case_history_cache,
):
    """
    Creates an investigation task in Jira for a subtask with unique failures.
    Groups failures by error, outputs a Jira-friendly description with a table of
    test names, components, duration, and RCA details (once).
    """
    # Group by error
    error_to_cases = group_errors_by_type(subtask_unique_failures)
    case_duration_lookup = build_case_duration_lookup(
        subtask_unique_failures, latest_build_id
    )

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
        (
            printed_rows,
            rca_info,
            batch_name,
            test_selector,
            github_compare,
            component_name,
        ) = build_case_rows(
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
        label=None,
    )

    print(f"✔ Created investigation task for subtask {subtask_id}: {issue.key}")
    return issue


def analyze_testflow(builds):
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

    jira_connection = get_jira_connection()

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
