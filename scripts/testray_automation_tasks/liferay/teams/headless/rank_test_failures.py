#!/usr/bin/env python3
import sys
import os
from typing import Any, Dict, Set

# Ensure repo root is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from collections import defaultdict
from dateutil.parser import parse as parse_date

from utils.liferay_utils.testray_utils.testray_api import (
    get_routine_to_builds,
    get_all_build_case_results,
    get_all_cases_info_from_build,   # bulk case meta per build
    get_component_name,              # lazy resolve for top-N
    fetch_case_results,              # case history (has PASSED + FAILED etc.)
    HEADLESS_ROUTINE_ID,
)

# Non-passing statuses
BAD_STATUSES = {"FAILED", "BLOCKED", "TESTFIX"}
# Count these as "a run"
RUN_STATUSES = BAD_STATUSES | {"PASSED"}  # ignore setup/skipped/etc.

# Months: May–Aug
INCLUDED_MONTHS = {9,10}

# Skip any CASE whose name contains these substrings
IGNORE_CASE_SUBSTRINGS = ("PortalLogAssertorTest-modules", "Top Level Build")


def _build_case_meta_for_build(build_id):
    """
    Bulk-load case metadata for a build:
      { case_id -> {"name": str, "component_id": int or None} }
    """
    meta = {}
    for item in get_all_cases_info_from_build(build_id):  # paginated internally
        case = item.get("r_caseToCaseResult_c_case") or {}
        case_id = case.get("id") or item.get("r_caseToCaseResult_c_caseId")
        if not case_id:
            continue
        meta[case_id] = {
            "name": case.get("name", f"Case {case_id}"),
            "component_id": case.get("r_componentToCases_c_componentId"),
        }
    return meta


def _in_range(d, start_d, end_d):
    """Compare by calendar date to avoid tz headaches."""
    return start_d <= d.date() <= end_d



def collect_case_failures_for_year(year: int = 2025) -> Dict[int, Dict[str, Any]]:
    """
    Count FAILS per test case across Headless builds in selected months of {year}.
    Assume every test runs in each analyzed build, so RUNS = number of builds.
    Also collect any linked issues for each case across builds.
    """
    print(f"📊 Collecting HEADLESS test case results for {year} (months {sorted(INCLUDED_MONTHS)})...")

    builds = get_routine_to_builds()  # already Headless only
    case_stats: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
        "runs": 0,
        "fails": 0,
        "name": None,
        "component_id": None,
        "issues": set(),      # type: Set[str]
        "issues_str": "",     # flattened string for printing
    })

    analyzed_builds: list[int] = []

    # --- Pass 1: gather failures, metadata, and issues ---
    for build in builds:
        due_str = build.get("dueDate")
        if not due_str:
            continue

        dt = parse_date(due_str)

        # only keep builds in correct year + included months
        if dt.year != year or dt.month not in INCLUDED_MONTHS:
            continue

        build_id = build.get("id")
        if not build_id:
            continue

        analyzed_builds.append(build_id)
        print(f"🔍 Processing build {build.get('name', '')} ({build_id})")

        # Bulk metadata for this build
        case_meta = _build_case_meta_for_build(build_id)

        # Get all case results (usually only failing ones)
        for result in get_all_build_case_results(build_id):
            case_id = result.get("r_caseToCaseResult_c_caseId")
            if not case_id:
                continue

            meta = case_meta.get(case_id)
            if not meta:
                continue

            case_name = meta["name"] or f"Case {case_id}"
            if any(sub in case_name for sub in IGNORE_CASE_SUBSTRINGS):
                continue

            status_obj = result.get("dueStatus")
            status = status_obj.get("key") if isinstance(status_obj, dict) else status_obj
            if not status:
                continue

            stats = case_stats[case_id]
            if stats["name"] is None:
                stats["name"] = case_name
                stats["component_id"] = meta["component_id"]

            # Track fails
            if status in BAD_STATUSES:
                stats["fails"] += 1

            # --- Track issues safely ---
            issues_str = result.get("issues")
            if issues_str:
                for issue in issues_str.split(","):
                    issue = issue.strip()
                    if issue:
                        stats["issues"].add(issue)

    # --- Pass 2: normalize runs and flatten issues ---
    total_runs = len(analyzed_builds)
    print(f"📈 Normalizing RUNS to {total_runs} builds for every case")

    for stats in case_stats.values():
        stats["runs"] = total_runs
        if isinstance(stats.get("issues"), set):
            # Put the stringified version in a separate field
            stats["issues_str"] = ", ".join(sorted(stats["issues"]))

    return case_stats

def rank_worst_cases(case_stats, top_n=50, min_runs=3):
    """Rank by highest fail ratio; break ties by fails then runs; ignore low-sample cases."""
    ranked = []
    for case_id, stats in case_stats.items():
        runs = stats["runs"]
        fails = stats["fails"]
        if runs < min_runs:
            continue
        fail_ratio = fails / runs if runs else 0.0
        ranked.append({
            "case_id": case_id,
            "name": stats["name"],
            "component_id": stats["component_id"],
            "runs": runs,
            "fails": fails,
            "fail_ratio": fail_ratio,
            "issues": stats.get("issues_str", ""),
        })
    ranked.sort(key=lambda x: (-x["fail_ratio"], -x["fails"], -x["runs"]))
    return ranked[:top_n]


def print_ranking(ranked_cases):
    # Lazily resolve component names only for the top results
    component_cache = {}

    def comp_name(cid):
        if not cid:
            return "Unknown"
        if cid not in component_cache:
            try:
                component_cache[cid] = get_component_name(cid)
            except Exception:
                component_cache[cid] = f"Component {cid}"
        return component_cache[cid]

    print("\n--- Worst Failing Tests Ranking ---\n")
    header = f"{'Case ID':<10} {'Fails':<6} {'Runs':<6} {'Fail %':<8} {'Component':<30} {'Issues':<25} Name"
    print(header)
    print("-" * len(header))

    for case in ranked_cases:
        fail_pct = f"{case['fail_ratio']*100:.1f}%"
        component = comp_name(case["component_id"])
        issues = (case.get("issues") or "")
        print(f"{fail_pct:<8} {component:<30} {issues}")



if __name__ == "__main__":
    stats = collect_case_failures_for_year(2025)
    ranked = rank_worst_cases(stats, top_n=50, min_runs=3)
    print_ranking(ranked)