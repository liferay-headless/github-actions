import base64
import requests
from functools import lru_cache
import os
from typing import Optional

BASE_URL = "https://testray.liferay.com/o/c"
TESTRAY_REST_URL = "https://testray.liferay.com/o/testray-rest/v1.0"
HEADLESS_ROUTINE_ID = 994140
ACCEPTANCE_ROUTINE_ID = 590307
STATUS_FAILED_BLOCKED_TESTFIX = "FAILED,TESTFIX,BLOCKED"


def assign_issue_to_case_result_batch(batch_updates):
    """Update a batch of case results with issues and due statuses."""
    for item in batch_updates:
        case_result_id = item["id"]
        payload = {"dueStatus": item["dueStatus"], "issues": item["issues"]}
        url = f"{BASE_URL}/caseresults/{case_result_id}"
        _put_json(url, payload)


def autofill_build(testray_build_id_1, testray_build_id_2):
    """Trigger autofill between two Testray builds."""
    url = f"{TESTRAY_REST_URL}/testray-build-autofill/{testray_build_id_1}/{testray_build_id_2}"
    response = requests.post(url, headers=_get_headers(), data="")
    response.raise_for_status()
    return response.json()


def complete_task(task_id):
    url = f"{BASE_URL}/tasks/{task_id}"
    payload = {"dueStatus": {"key": "COMPLETE", "name": "Complete"}}
    response = requests.patch(
        url,
        json=payload,
        headers={**_get_headers(), "Content-Type": "application/json"},
    )
    response.raise_for_status()
    return response.json()


def create_task(build):
    """Create a task for a build."""
    payload = {
        "name": build["name"],
        "r_buildToTasks_c_buildId": build["id"],
        "dueStatus": {"key": "INANALYSIS", "name": "In Analysis"},
    }
    response = requests.post(
        f"{BASE_URL}/tasks/",
        json=payload,
        headers={**_get_headers(), "Content-Type": "application/json"},
    )
    response.raise_for_status()
    return response.json()


def create_testflow(task_id):
    """Create testflow for a task."""
    url = f"{TESTRAY_REST_URL}/testray-testflow/{task_id}"
    response = requests.post(url, headers=_get_headers(), data="")
    response.raise_for_status()
    return response.json()


def fetch_case_results(case_id, routine_id, status=None, page_size=500):
    base_url = f"{TESTRAY_REST_URL}/testray-case-result-history/{case_id}"
    page = 1
    all_items = []

    while True:
        params = (
            f"testrayRoutineIds={routine_id}"
            + (f"&status={status}" if status else "")
            + f"&page={page}&pageSize={page_size}"
        )
        url = f"{base_url}?{params}"
        result = _get_json(url)
        items = result.get("items", [])
        all_items.extend(items)

        if len(items) < page_size:
            break
        page += 1

    return all_items

def get_all_build_case_results(build_id):
    """Fetch all case results for a given build (paginated)."""
    page = 1
    all_items = []

    while True:
        url = f"{BASE_URL}/builds/{build_id}/buildToCaseResult?pageSize=500&page={page}"
        data = _get_json(url)
        items = data.get("items", [])
        all_items.extend(items)

        if len(items) < 500:
            break
        page += 1

    return all_items


@lru_cache(maxsize=None)
def get_build_info(build_id):
    """Get build metadata, including routine ID and due date."""
    url = f"{BASE_URL}/builds/{build_id}?fields=dueDate,gitHash,name,id,importStatus,r_routineToBuilds_c_routineId&nestedFields=buildToTasks"
    return _get_json(url)


def get_build_tasks(build_id):
    """Get tasks associated with a build."""
    url = f"{BASE_URL}/builds/{build_id}/buildToTasks?fields=id,dueStatus"
    return _get_json(url).get("items", [])


@lru_cache(maxsize=None)
def get_case_info(case_id):
    """Get the name and priority of a test case."""
    url = f"{BASE_URL}/cases/{case_id}"
    return _get_json(url)


def get_case_result(case_result_id):
    url = f"{BASE_URL}/caseresults/{case_result_id}"
    return _get_json(url)


def get_case_count_by_type_in_build(build_id, case_type_id):
    """Get the count of unique cases of a specific type that have results in a given build."""
    if not case_type_id:
        return 0

    all_items = []
    page = 1
    page_size = 500

    while True:
        url = (
            f"{BASE_URL}/builds/{build_id}/buildToCaseResult"
            f"?nestedFields=r_caseToCaseResult_c_case&pageSize={page_size}&page={page}"
        )
        data = _get_json(url)
        items = data.get("items", [])
        all_items.extend(items)

        if len(items) < page_size:
            break
        page += 1

    matching_case_ids = {
        item.get("r_caseToCaseResult_c_caseId")
        for item in all_items
        if item.get("r_caseToCaseResult_c_case", {}).get(
            "r_caseTypeToCases_c_caseTypeId"
        )
        == case_type_id
    }

    return len(matching_case_ids)


@lru_cache(maxsize=None)
def get_case_type_id_by_name(case_type_name):
    """Get the ID of a case type by its name."""
    url = f"{BASE_URL}/casetypes?filter=name eq '{case_type_name}'&fields=id"
    result = _get_json(url)
    items = result.get("items", [])
    if items:
        return items[0].get("id")
    return None


@lru_cache(maxsize=None)
def get_case_type_name(case_type_id):
    """Get name of a case type by ID."""
    url = f"{BASE_URL}/casetypes/{case_type_id}?fields=name"
    return _get_json(url).get("name", "Unknown")


@lru_cache(maxsize=None)
def get_component_name(component_id):
    """Get name of a component by ID."""
    url = f"{BASE_URL}/components/{component_id}?fields=name"
    return _get_json(url).get("name", "Unknown")


def get_routine_to_builds(routine_id):
    """Fetch all builds for a routine, remove pagination and sort by dateCreated descending."""
    url = f"{BASE_URL}/routines/{routine_id}/routineToBuilds?fields=dueDate,name,id,importStatus,gitHash,r_routineToBuilds_c_routineId,dateCreated&pageSize=-1"
    items = _get_json(url).get("items", [])
    return sorted(items, key=lambda b: b.get("dateCreated", ""), reverse=True)

def get_build_sha(build_id):
    """Get gitHash of a specific build."""
    url = f"{BASE_URL}/builds/{build_id}?fields=gitHash"
    return _get_json(url).get("gitHash")

def get_build_metrics(routine_id):
    """
    Fetch all acceptance builds with their metrics from Testray REST API.
    Returns a list of items with fields like:
      - testrayBuildGitHash
      - testrayBuildId
      - testrayBuildName
      - testrayStatusMetric (dict of counts)
    """
    url = f"{TESTRAY_REST_URL}/testray-status-metrics/by-testray-routineId/{routine_id}/testray-builds-metrics"
    data = _get_json(url)
    return data.get("items", [])

def get_subtask_case_results(subtask_id):
    """Get case results under a subtask."""
    url = f"{BASE_URL}/subtasks/{subtask_id}/subtaskToCaseResults?fields=id,executionDate,errors,issues,r_caseToCaseResult_c_caseId,r_componentToCaseResult_c_componentId&pageSize=-1"
    return _get_json(url).get("items", [])


def get_task_status(task_id):
    """Get the status of a task."""
    url = f"{BASE_URL}/tasks/{task_id}?fields=dueStatus"
    return _get_json(url)


def get_task_subtasks(task_id):
    """Get subtasks associated with a task."""
    url = f"{BASE_URL}/tasks/{task_id}/taskToSubtasks?pageSize=-1"
    return _get_json(url).get("items", [])


def update_subtask_status(subtask_id: str, issues: Optional[str] = None) -> None:
    """Mark a subtask as complete."""
    url = f"{BASE_URL}/subtasks/{subtask_id}"
    payload = {"dueStatus": {"key": "COMPLETE", "name": "Complete"}}
    if issues:
        payload["issues"] = issues
    _put_json(url, payload)
    print(f"Subtask {subtask_id} marked as COMPLETE.")


def _get_json(url):
    """Send GET request and return JSON response. Refresh token if 401."""
    response = requests.get(url, headers=_get_headers())

    if response.status_code == 401:
        _get_headers.cache_clear()
        response = requests.get(url, headers=_get_headers())

    response.raise_for_status()
    return response.json()


def _put_json(url, payload):
    """Send PUT request with JSON payload."""
    response = requests.put(
        url,
        json=payload,
        headers={**_get_headers(), "Content-Type": "application/json"},
    )
    response.raise_for_status()
    return response.json()


@lru_cache()
def _get_headers():
    TESTRAY_CLIENT_ID = os.getenv("TESTRAY_CLIENT_ID") or (_ for _ in ()).throw(
        EnvironmentError("TESTRAY_CLIENT_ID environment variable is not set.")
    )
    TESTRAY_CLIENT_SECRET = os.getenv("TESTRAY_CLIENT_SECRET") or (_ for _ in ()).throw(
        EnvironmentError("TESTRAY_CLIENT_SECRET environment variable is not set.")
    )
    response = requests.post(
        "https://testray.liferay.com/o/oauth2/token",
        headers={
            "Authorization": f"Basic {base64.b64encode(f'{TESTRAY_CLIENT_ID}:{TESTRAY_CLIENT_SECRET}'.encode()).decode()}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"grant_type": "client_credentials"},
    )
    response.raise_for_status()
    return {
        "Authorization": f"Bearer {response.json()['access_token']}",
        "Accept": "application/json",
    }
