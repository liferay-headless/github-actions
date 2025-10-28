#!/usr/bin/env python
import os
import sys
import json

sys.path.append(os.path.join(os.path.join(sys.path[0], '..', '..', '..'), 'utils'))

from liferay.teams.headless.headless_contstants import Filter, HeadlessStrings
from utils.liferay_utils.jira_utils.jira_constants import Status, Strings
from utils.liferay_utils.jira_utils.jira_helpers import create_poshi_automation_task_for, \
    close_functional_automation_subtask, create_investigation_task_for
from utils.liferay_utils.jira_utils.jira_liferay import get_jira_connection


def _create_poshi_task_for(jira_local, parent_story, poshi_automation_table):
    parent_key = parent_story.key
    parent_summary = parent_story.get_field('summary')
    print("Creating poshi automation task for story", parent_key)
    summary = 'Product QA | Functional Automation - ' + parent_key + ' - ' + parent_summary

    description = HeadlessStrings.poshi_task_description + poshi_automation_table
    new_issue = create_poshi_automation_task_for(jira_local, parent_story, summary, description)

    if new_issue:
        print("Poshi task with summary", summary, " already existed for", parent_key)
    else:
        print("Poshi task ", new_issue.key, " created for", parent_key)
    return new_issue

def create_investigation_task(jira_local):
    """
    Create or find an investigation task in JIRA based on Testray test results.

    :param jira: JIRA client instance for issue management.
    :param json_file_path: Path to the JSON file containing Testray test results.
    :return: List of created or found JIRA issues.
    """
    json_file_path = "/home/me/Projects/eng/testray_json/failed-tests.json"

    with open(json_file_path, "r") as file:
        data = json.load(file)

    test_result_items = data.get("items", [])
    created_issues = []

    # Search existing investigation tasks
    investigation_tasks = jira_local.search_issues(Filter.Investigation_Testing_Tasks)

    for test_result in test_result_items:
        testray_case_name = test_result["testrayCaseName"]
        testray_component_name = test_result["testrayComponentName"]
        testray_case_type_name = test_result["testrayCaseTypeName"]
        testray_run_name = test_result["testrayRunName"]
        error = test_result["error"]
        testray_case_result_id = test_result["testrayCaseResultId"]

        # Check if an investigation task already exists for this case
        task_exists = False
        for task in investigation_tasks:
            summary = task.fields.summary
            if testray_case_name in summary:
                print(
                    f"Testing task to investigate failure in {testray_case_name} "
                    f"already exists with key {task.key}"
                )
                task_exists = True
                break

        if task_exists:
            continue

        # Create the summary and description for the new task
        summary = f"[HL] Investigate failure in {testray_case_name}"
        description = (
            f"### Summary: [HL] Investigate failure in {testray_case_name}\n"
            f"**Component**: {testray_component_name}\n\n"
            f"**Test type**: {testray_case_type_name}\n"
            f"**Environment**: {testray_run_name}\n"
            f"**Error**: {error}\n\n"
            f"**Testray link**: "
            f"https://testray.liferay.com/#/project/35392/routines/994140/build/78629312/case-result/{testray_case_result_id}"
        )
        component = testray_component_name
        environment = testray_run_name

        # Create the new issue
        new_issue = create_investigation_task_for(jira_local,summary,description,component,environment)
        created_issues.append(new_issue)

    return created_issues

def update_creation_subtask(jira):
    print("Updating test creation subtasks for Headless team...")
    stories_with_test_creation_subtask = \
        jira.search_issues(Filter.Integration_In_Development_Sub_task_creation_Headless_team)
    for story in stories_with_test_creation_subtask:
        for subtask in story.fields.subtasks:
            summary = subtask.fields.summary
            key = subtask.key
            assignee = jira.issue(key, fields='assignee')
            if summary == Strings.subtask_test_creation_summary:
                print("Updating "+key+" ...")
                if subtask.fields.status.name == Status.Open:
                    description = HeadlessStrings.test_creation_description
                    subtask.update(fields={'description': description})
                    if assignee != 'Support QA':
                        jira.assign_issue(subtask.id, 'support-qa')
                    break
    print("Subtasks Test Creation for Headless team are up to date")


def update_validation_subtask(jira):
    print("Updating test validation subtasks for Headless team...")
    stories_with_test_validation_subtask = jira.search_issues(Filter.Product_QA_Test_Validation_Round_1)
    for story in stories_with_test_validation_subtask:
        for subtask in story.fields.subtasks:
            summary = subtask.fields.summary
            key = subtask.key
            assignee = jira.issue(key, fields='assignee')
            if 'Product QA | Test Validation' in summary:
                if subtask.fields.status.name == Status.Open:
                    print("Updating "+key+" ...")
                    description = HeadlessStrings.test_validation_round_1_description
                    subtask.update(fields={'description': description})
                    if assignee != 'Support QA':
                        jira.assign_issue(subtask.id, 'support-qa')
                    break
    print("Validation subtasks for Headless team are up to date")


def create_poshi_automation_task(jira):
    print("Creating Poshi tasks...")
    output_message = ''
    stories_without_poshi_automation_created = \
        jira.search_issues(Filter.Headless_Team_Ready_to_create_POSHI_Automation_Task)
    for story in stories_without_poshi_automation_created:
        for subtask in story.get_field('subtasks'):
            if subtask.fields.summary == Strings.subtask_test_creation_summary:
                description = jira.issue(subtask.id, fields='description').fields.description
                table_starring_string = ''
                if description is not None:
                    if description.find('||*Requirement*||') != -1:
                        table_starring_string = '||*Requirement*||'
                    elif description.find('||Requirement||') != -1:
                        table_starring_string = '||Requirement||'
                    table_staring_position = description.find(table_starring_string)
                    table_ending_string = ''
                    if description.find('*Exploratory') != -1:
                        table_ending_string = '*Exploratory'
                    elif description.find('Exploratory') != -1:
                        table_ending_string = 'Exploratory'
                    table_ending_position = description.find(table_ending_string)
                    poshi_automation_table = description[table_staring_position:table_ending_position - 1]
                    poshi_task = _create_poshi_task_for(jira, story, poshi_automation_table)
                    if poshi_task:
                        close_functional_automation_subtask(jira, story, poshi_task.key)
                        output_message += "Functional Automation created for " + story.get_field('summary')
                    else:
                        output_message += "Functional Automation already exists for " + story.get_field('summary')
    print(output_message)


if __name__ == "__main__":
    jira_connection = get_jira_connection()
    create_investigation_task(jira_connection)
    jira_connection.close()
