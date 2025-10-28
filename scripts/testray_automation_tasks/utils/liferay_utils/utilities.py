#!/usr/bin/env python3

from datetime import datetime
import re

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
    month = today.month
    year = today.year

    if month <= 3:
        start_month = 1
        quarter_number = 1
    elif month <= 6:
        start_month = 4
        quarter_number = 2
    elif month <= 9:
        start_month = 7
        quarter_number = 3
    else:
        start_month = 10
        quarter_number = 4

    quarter_start = datetime(year, start_month, 1).date()
    return quarter_start, quarter_number, year

def normalize_error(error):
    """Normalize and clean error messages for comparison and pattern matching."""
    if not error:
        return ""

    # Collapse whitespace
    error = ' '.join(error.strip().split())

    # Remove timestamps, memory addresses, test durations, or dynamic values
    error = re.sub(r'\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}', '', error)  # timestamps
    error = re.sub(r'0x[0-9A-Fa-f]+', '', error)  # memory addresses
    error = re.sub(r'\d+\s*(ms|s|seconds|minutes|m)', '', error)  # durations
    error = re.sub(r'".*?"', '"..."', error)  # replace quoted strings with placeholder

    return error

def parse_execution_date(date_str):
    date_str = date_str.strip()

    # Remove 'Z' if present (which means UTC in ISO 8601)
    if date_str.endswith('Z'):
        date_str = date_str[:-1]

    # Replace 'T' with space to normalize the format
    date_str = date_str.replace('T', ' ')

    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None