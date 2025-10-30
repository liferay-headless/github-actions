from jira import JIRA
import os


def get_jira_connection():
    url = "https://liferay.atlassian.net"
    user = os.getenv("JIRA_API_USER") or (_ for _ in ()).throw(
        EnvironmentError("JIRA_API_USER environment variable is not set.")
    )
    token = os.getenv("JIRA_API_TOKEN") or (_ for _ in ()).throw(
        EnvironmentError("JIRA_API_TOKEN environment variable is not set.")
    )
    print(f"Connecting to Jira in URL {url} with user {user}")
    jira = JIRA(url, basic_auth=(user, token))
    print("Connected to Jira")
    return jira


def close_issue(jira_local, issue_key, build_hash):
    """
    Workflow to close an issue properly:
    1. Move parent to 'Selected for Development'.
    2. Close all sub-tasks (created automatically once parent is in SFD).
    3. Close parent with 'Discarded' resolution.
    """
    try:
        parent_issue = jira_local.issue(issue_key)

        # --- Step 1: Parent → 'Selected for Development'
        transitions = jira_local.transitions(issue_key)
        selected_dev_transition = next(
            (t for t in transitions if t["name"] == "Selected for Development"), None
        )
        if selected_dev_transition:
            jira_local.transition_issue(
                issue_key, transition=selected_dev_transition["id"]
            )
            print(f"✔ {issue_key} → 'Selected for Development'")
        else:
            print(
                f"✘ Could not find 'Selected for Development' transition for {issue_key}"
            )
            return

        # --- Step 2: Close all sub-tasks (after automation creates them)
        parent_issue = jira_local.issue(issue_key)  # refresh to load subtasks
        subtasks = getattr(parent_issue.fields, "subtasks", [])
        for subtask in subtasks:
            subtask_key = subtask.key
            print(f"→ Closing child {subtask_key}")
            _transition_to_closed(jira_local, subtask_key, build_hash)

        # --- Step 3: Parent → 'Closed' with resolution 'Discarded'
        transitions = jira_local.transitions(issue_key)
        close_transition = next((t for t in transitions if t["name"] == "Closed"), None)
        if close_transition:
            jira_local.transition_issue(
                issue_key,
                transition=close_transition["id"],
                resolution={"name": "Discarded"},
            )
            jira_local.add_comment(
                issue_key, f"Closing. Not reproducible in current SHA {build_hash}"
            )
            print(f"✔ {issue_key} → 'Closed' with 'Discarded'")
        else:
            print(f"✘ Could not find 'Closed' transition for {issue_key}")

    except Exception as e:
        print(f"✘ Failed to process issue {issue_key}: {e}")


def _transition_to_closed(jira_local, issue_key, build_hash):
    """
    Closes a single sub-task (directly to 'Closed' with 'Discarded').
    """
    try:
        transitions = jira_local.transitions(issue_key)
        close_transition = next((t for t in transitions if t["name"] == "Closed"), None)

        if close_transition:
            jira_local.transition_issue(
                issue_key,
                transition=close_transition["id"],
                resolution={"name": "Discarded"},
            )
            jira_local.add_comment(
                issue_key,
                f"Closing sub-task. Not reproducible in current SHA {build_hash}",
            )
            print(f"✔ {issue_key} → 'Closed' with 'Discarded'")
        else:
            print(f"✘ Could not find 'Closed' transition for sub-task {issue_key}")

    except Exception as e:
        print(f"✘ Failed to close sub-task {issue_key}: {e}")


def create_investigation_task_for(
    jira_local, summary, description, component, environment
):
    issue_dict = {
        "project": {"key": "LPD"},
        "summary": summary,
        "description": description,
        "issuetype": {"name": "Task"},
        "components": [{"name": component}],
        "labels": ["hl_routine_tasks"],
        "customfield_environment": environment,
    }
    new_issue = jira_local.create_issue(fields=issue_dict)
    print(f"Created new investigation task: {new_issue.key}")
    return new_issue


def create_jira_task(jira_local, epic, summary, description, component, label):
    """
    Creates a Jira investigation task for unique failures.
    `component` can be:
      - a single string
      - a list of strings
      - a list of dicts
      - None (defaults to no components)
    """

    # Normalize components into the correct format
    components_list = []
    if isinstance(component, str):
        components_list = [{"name": component}]
    elif isinstance(component, list):
        if all(isinstance(c, dict) for c in component):
            components_list = component
        else:
            components_list = [{"name": str(c)} for c in component]
    elif component is None:
        components_list = []
    else:
        raise TypeError(
            f"Invalid type for component: {type(component).__name__}. "
            "Must be str, list, or None."
        )

    issue_dict = {
        "project": {"key": "LPD"},
        "summary": summary,
        "description": description,
        "parent": {"id": epic.id},
        "issuetype": {"name": "Task"},
        "components": components_list,
    }

    new_issue = jira_local.create_issue(fields=issue_dict)

    jira_local.issue(new_issue.key).update(
        update={"labels": [{"add": "hl_routine_tasks"}]}
    )
    if label:
        jira_local.issue(new_issue.key).update(update={"labels": [{"add": label}]})

    return new_issue


def get_all_issues(jira_local, jql_str, fields):
    issues = []
    i = 0
    chunk_size = 50
    while True:
        chunk = jira_local.search_issues(
            jql_str, startAt=i, maxResults=chunk_size, fields=fields
        )
        i += chunk_size
        issues += chunk.iterable
        if i >= chunk.total:
            break
    return issues


def get_issue_status_by_key(jira_local, issue_key):
    """
    Retrieves the issue by key and returns its status name.

    :param jira_local: Authenticated Jira connection.
    :param issue_key: The key of the issue (e.g., "LPD-12345").
    :return: Tuple (issue, status_name) if found, else (None, None)
    """
    try:
        issue = jira_local.issue(issue_key, fields="status")
        return issue, issue.fields.status.name
    except Exception as e:
        print(f"Error retrieving issue {issue_key}: {str(e)}")
        return None, None
