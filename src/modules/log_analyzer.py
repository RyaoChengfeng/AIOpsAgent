"""
Log Analysis Module
Provides utility functions and LangChain tools for parsing and analyzing log files.
"""

import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from langchain.chains import llm
from langchain.tools import BaseTool
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field

from config.settings import get_config, Settings
from src.utils.logger import get_logger
from src.utils.helpers import (
    parse_log_level, extract_error_patterns, safe_json_loads,
    truncate_string, run_command, format_duration
)
from src.utils.exceptions import LogAnalysisError, FileOperationError

logger = get_logger(__name__)


class LogAction(BaseModel):
    """Single log operation parsing model"""
    action: str = Field(description="Operation type: analyze_file, analyze_errors, analyze_warnings, generate_report, search_log, search_all_logs, get_statistics")
    file_path: Optional[str] = Field(default=None, description="Target log file path (e.g., /var/log/nginx/error.log)")
    keyword: Optional[str] = Field(default=None, description="Keyword for search operations")
    time_filter: Optional[str] = Field(default=None, description="Time range description (e.g., '1 hour', '2 days', 'today', 'yesterday')")


class LogActionMultiple(BaseModel):
    """Supports multiple actions in one request"""
    actions: List[LogAction] = Field(description="List of log actions to execute in order")


log_parser = PydanticOutputParser(pydantic_object=LogActionMultiple)

log_prompt = PromptTemplate(
    template="""You are a professional System Log Analysis expert. 
Strictly extract structured information from the following natural language instruction:

The user may request multiple actions in order. Each action must be one of:
- analyze_file
- analyze_errors
- analyze_warnings
- generate_report
- search_log
- search_all_logs
- get_statistics

Each action should include optional fields: file_path, keyword, time_filter.

Command: {command}

{format_instructions}""",
    input_variables=["command"],
    partial_variables={"format_instructions": log_parser.get_format_instructions()},
)


class LogAnalyzerConfig(BaseModel):
    """Log analyzer configuration model"""
    max_file_size: str = Field(default_factory=lambda: get_config('file_manager.max_file_size', '100MB'))
    error_patterns: List[str] = Field(default_factory=lambda: [
        'error', 'exception', 'failed', 'timeout', 'connection refused', 'permission denied'
    ])
    warning_patterns: List[str] = Field(default_factory=lambda: [
        'warning', 'deprecated', 'notice'
    ])
    time_formats: List[str] = Field(default_factory=lambda: [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%b %d %H:%M:%S",
        "%d/%b/%Y:%H:%M:%S"
    ])


class LogAnalyzerTool(BaseTool):
    """Log Analysis LangChain Tool (supports multiple actions)"""

    name: str = "log_analyzer"
    description: str = (
        "A tool for analyzing log files. Supports parsing service logs, searching for errors/exceptions, "
        "generating reports, and calculating statistics. "
        "Supports multiple actions in one request and time-based analysis "
        "(e.g., 'analyze errors from the last 1 hour and then generate report')."
    )
    args_schema: Optional[BaseModel] = None

    def __init__(self):
        """Initialize Log Analyzer tool"""
        super().__init__()

    def _parse_command(self, command: str) -> LogActionMultiple:
        """
        Parse log analysis command using AI to extract multiple actions.
        Returns LogActionMultiple containing a list of LogAction objects.
        """
        settings = Settings()
        openai_config = settings.get_openai_config()
        max_tokens = openai_config.get('max_tokens', 2000)
        model = openai_config.get('model', 'gpt-3.5-turbo')
        temperature = openai_config.get('temperature', 0)
        api_key = openai_config.get('api_key')
        base_url = openai_config.get('base_url')

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=base_url,
            max_tokens=max_tokens,
            default_headers={
                "HTTP-Referer": "https://localhost/",
                "X-Title": "DevOps-AIOps-Agent"
            }
        )

        chain = log_prompt | llm | log_parser
        try:
            return chain.invoke({"command": command})
        except Exception as e:
            logger.warning(f"AI command parsing failed: {e}")
            # fallback: single unknown action
            return LogActionMultiple(actions=[LogAction(action="unknown")])

    def _run(self, command: str) -> str:
        """
        Execute one or more log analysis operations in order.
        """
        try:
            parsed_commands = self._parse_command(command)
            results = []

            for action_obj in parsed_commands.actions:
                action = action_obj.action
                log_file = action_obj.file_path
                keyword = action_obj.keyword
                time_str = action_obj.time_filter
                time_range = self._parse_time_filter(time_str) if time_str else None

                # 调用原来的功能函数
                if action == "analyze_file":
                    if not log_file:
                        results.append("Please specify a log file path.")
                        continue
                    results.append(self._analyze_log_file(log_file, time_range))

                elif action == "analyze_errors":
                    if not log_file:
                        results.append("Please specify a log file path.")
                        continue
                    results.append(self._analyze_errors(log_file, time_range))

                elif action == "analyze_warnings":
                    if not log_file:
                        results.append("Please specify a log file path.")
                        continue
                    results.append(self._analyze_warnings(log_file, time_range))

                elif action == "generate_report":
                    if not log_file:
                        results.append("Please specify a log file path.")
                        continue
                    results.append(self._generate_report(log_file, time_range))

                elif action == "search_log":
                    if not log_file:
                        results.append("Please specify a log file path.")
                        continue
                    if not keyword:
                        results.append("Please specify a keyword to search.")
                        continue
                    results.append(self._search_log(log_file, keyword, time_range))

                elif action == "search_all_logs":
                    if not keyword:
                        results.append("Please specify a keyword to search.")
                        continue
                    results.append(self._search_all_logs(keyword, time_range))

                elif action == "get_statistics":
                    if not log_file:
                        results.append("Please specify a log file path.")
                        continue
                    results.append(self._get_log_statistics(log_file, time_range))

                else:
                    results.append(f"Unsupported log operation: {action}")

            return "\n\n".join(results)

        except Exception as e:
            logger.error(f"Log analysis failed: {e}")
            raise LogAnalysisError(f"Log analysis execution failed: {str(e)}")

    def _parse_time_filter(self, time_str: str) -> Optional[Tuple[datetime, datetime]]:
        """
        Parse natural language time string into start/end datetime.
        Supported formats: "1 hour", "2 days", "30 mins", "today", "yesterday"
        """
        try:
            time_str = time_str.lower()
            end_time = datetime.now()
            start_time = None

            if "today" in time_str:
                start_time = datetime(end_time.year, end_time.month, end_time.day)

            elif "yesterday" in time_str:
                yesterday = end_time - timedelta(days=1)
                start_time = datetime(yesterday.year, yesterday.month, yesterday.day)
                end_time = datetime(end_time.year, end_time.month, end_time.day)

            else:
                # Try to extract number and unit
                match = re.search(r'(\d+)\s*(hour|hr|day|d|min|minute|sec|second)', time_str)
                if match:
                    num = int(match.group(1))
                    unit = match.group(2)

                    if 'hour' in unit or 'hr' in unit:
                        start_time = end_time - timedelta(hours=num)
                    elif 'day' in unit or 'd' in unit:
                        start_time = end_time - timedelta(days=num)
                    elif 'min' in unit:
                        start_time = end_time - timedelta(minutes=num)
                    elif 'sec' in unit:
                        start_time = end_time - timedelta(seconds=num)

            return (start_time, end_time) if start_time else None
        except Exception as e:
            logger.warning(f"Failed to parse time filter '{time_str}': {e}")
            return None

    def _parse_log_time(self, line: str) -> Optional[datetime]:
        """Parse timestamp from a log line"""
        config = LogAnalyzerConfig()
        for fmt in config.time_formats:
            try:
                # Try extracting time from the beginning of the line
                time_str = line[:25].strip()
                # Clean up brackets if present
                time_str = time_str.replace('[', '').replace(']', '')
                # Attempt strictly or via substring match could be complex,
                # here we try exact format match on truncated string
                return datetime.strptime(time_str.split('+')[0].strip(), fmt)
            except ValueError:
                continue

        # Try JSON log format
        json_data = safe_json_loads(line)
        if json_data and isinstance(json_data, dict):
            for key in ['time', 'timestamp', 'log_time']:
                if key in json_data:
                    try:
                        return datetime.fromisoformat(str(json_data[key]).replace('Z', '+00:00'))
                    except ValueError:
                        continue
        return None

    def _filter_logs_by_time(self, content: str, time_range: Tuple[datetime, datetime]) -> str:
        """Filter log content by time range"""
        start_time, end_time = time_range
        filtered_lines = []

        for line in content.splitlines():
            log_time = self._parse_log_time(line)
            # If we can't parse time, we might keep it or drop it.
            # Usually strict filtering drops it, or keeps it if it looks like a continuation (stack trace).
            # For simplicity, we filter strictly on lines with timestamps here.
            if log_time:
                if start_time <= log_time <= end_time:
                    filtered_lines.append(line)
            # Optional: Add logic to include stack traces following a valid timestamped line

        return '\n'.join(filtered_lines)

    def _analyze_log_file(self, log_file: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """Analyze log file (supports time range)"""
        try:
            if not self._validate_log_file(log_file):
                return f"Cannot access log file '{log_file}' or file is too large."

            content = self._read_log_file(log_file)
            if not content:
                return f"Log file '{log_file}' is empty or unreadable."

            # Apply time filtering
            filtered_content = content
            time_filter_info = ""
            if time_range:
                filtered_content = self._filter_logs_by_time(content, time_range)
                time_filter_info = f"(Time Range: {time_range[0]} to {time_range[1]})"
                if not filtered_content:
                    return f"No log entries found in the specified time range {time_filter_info}"

            # Basic stats
            stats = parse_log_level(filtered_content)
            errors = extract_error_patterns(filtered_content)

            result = f"Log Analysis Report for '{log_file}' {time_filter_info}:\n"
            result += "=" * 50 + "\n"
            result += f"File Size: {self._get_file_size(log_file)}\n"
            result += f"Total Lines: {len(content.splitlines())}\n"
            result += f"Filtered Lines: {len(filtered_content.splitlines())}\n\n"

            result += "Log Level Statistics:\n"
            result += "-" * 20 + "\n"
            total = sum(stats.values())
            for level, count in stats.items():
                percentage = (count / total * 100) if total > 0 else 0
                result += f"{level}: {count} ({percentage:.1f}%)\n"

            if errors:
                result += f"\nFound {len(errors)} error patterns:\n"
                result += "-" * 20 + "\n"
                for error in errors[:10]:
                    result += f"- {truncate_string(error, 80)}\n"
                if len(errors) > 10:
                    result += f"\n... and {len(errors) - 10} more errors\n"
            else:
                result += "\n✅ No obvious error patterns found."

            # Recent logs
            lines = filtered_content.splitlines()
            recent_logs = '\n'.join(lines[-10:]) if lines else "No content"
            result += f"\nLast 10 Log Lines:\n"
            result += "-" * 20 + "\n"
            result += truncate_string(recent_logs, 1000)

            return result

        except Exception as e:
            logger.error(f"Analyze log failed: {e}")
            raise LogAnalysisError(f"Failed to analyze '{log_file}': {str(e)}")

    def _analyze_errors(self, log_file: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """Analyze error logs (supports time range and frequency stats)"""
        try:
            if not self._validate_log_file(log_file):
                return f"Cannot access log file '{log_file}'."

            content = self._read_log_file(log_file)

            filtered_content = content
            time_filter_info = ""
            if time_range:
                filtered_content = self._filter_logs_by_time(content, time_range)
                time_filter_info = f"(Time Range: {time_range[0]} to {time_range[1]})"
                if not filtered_content:
                    return f"No log entries found in the specified time range {time_filter_info}"

            errors = extract_error_patterns(filtered_content)

            if not errors:
                return f"✅ No errors found in '{log_file}' {time_filter_info}."

            # Categorize errors
            error_types = {
                'Connection Error': 0,
                'Permission Error': 0,
                'Timeout Error': 0,
                'File Error': 0,
                'Other Error': 0
            }

            error_timestamps = []
            for error in errors:
                error_lower = error.lower()
                if any(k in error_lower for k in ['connection', 'refused', 'timeout']):
                    error_types['Connection Error'] += 1
                elif any(k in error_lower for k in ['permission', 'access denied']):
                    error_types['Permission Error'] += 1
                elif 'timeout' in error_lower:
                    error_types['Timeout Error'] += 1
                elif any(k in error_lower for k in ['file', 'no such', 'cannot open']):
                    error_types['File Error'] += 1
                else:
                    error_types['Other Error'] += 1

                # Find timestamp for frequency analysis
                log_line = self._find_error_line(content, error)
                if log_line:
                    log_time = self._parse_log_time(log_line)
                    if log_time:
                        error_timestamps.append(log_time)

            # Frequency report
            frequency_report = ""
            if error_timestamps:
                error_timestamps.sort()
                hourly_counts = {}
                for dt in error_timestamps:
                    hour_key = dt.strftime("%Y-%m-%d %H:00")
                    hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1

                frequency_report = "\nError Frequency (Hourly):\n"
                frequency_report += "-" * 25 + "\n"
                recent_hours = sorted(hourly_counts.keys())[-6:]
                for hour in recent_hours:
                    frequency_report += f"{hour}: {hourly_counts[hour]} count(s)\n"

            critical_errors = [e for e in errors if any(kw in e.lower() for kw in ['fatal', 'critical', 'panic'])]
            critical_note = f"\n⚠️  Found {len(critical_errors)} critical errors!" if critical_errors else ""

            result = f"Error Analysis Report - '{log_file}' {time_filter_info}:\n"
            result += "=" * 40 + "\n"
            result += f"Total Errors: {len(errors)}\n{critical_note}\n\n"

            result += "Error Type Distribution:\n"
            result += "-" * 15 + "\n"
            total_errors = len(errors)
            for error_type, count in error_types.items():
                if count > 0:
                    percentage = (count / total_errors * 100)
                    result += f"{error_type}: {count} ({percentage:.1f}%)\n"

            result += frequency_report

            result += "\nTypical Examples (Top 5):\n"
            result += "-" * 20 + "\n"
            for error in errors[:5]:
                result += f"- {truncate_string(error, 100)}\n"

            if len(errors) > 5:
                result += f"\n... and {len(errors) - 5} more errors"

            return result

        except Exception as e:
            logger.error(f"Analyze errors failed: {e}")
            raise LogAnalysisError(f"Failed to analyze errors in '{log_file}': {str(e)}")

    def _analyze_warnings(self, log_file: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """Analyze warning logs"""
        try:
            if not self._validate_log_file(log_file):
                return f"Cannot access log file '{log_file}'."

            content = self._read_log_file(log_file)

            filtered_content = content
            time_filter_info = ""
            if time_range:
                filtered_content = self._filter_logs_by_time(content, time_range)
                time_filter_info = f"(Time Range: {time_range[0]} to {time_range[1]})"
                if not filtered_content:
                    return f"No log entries found in the specified time range {time_filter_info}"

            config = LogAnalyzerConfig()
            warning_pattern = re.compile('|'.join(config.warning_patterns), re.IGNORECASE)
            warnings = [line.strip() for line in filtered_content.splitlines() if warning_pattern.search(line)]

            if not warnings:
                return f"✅ No warnings found in '{log_file}' {time_filter_info}."

            result = f"Warning Analysis Report - '{log_file}' {time_filter_info}:\n"
            result += "=" * 40 + "\n"
            result += f"Total Warnings: {len(warnings)}\n\n"

            deprecation_warnings = [w for w in warnings if 'deprecated' in w.lower()]
            config_warnings = [w for w in warnings if 'config' in w.lower() or 'configuration' in w.lower()]
            other_warnings = [w for w in warnings if w not in deprecation_warnings and w not in config_warnings]

            result += "Warning Type Distribution:\n"
            result += "-" * 15 + "\n"
            # Prevent division by zero if len(warnings) is 0 (though handled by if check above)
            count = len(warnings)
            result += f"Deprecation: {len(deprecation_warnings)} ({len(deprecation_warnings)/count*100:.1f}%)\n"
            result += f"Configuration: {len(config_warnings)} ({len(config_warnings)/count*100:.1f}%)\n"
            result += f"Other: {len(other_warnings)} ({len(other_warnings)/count*100:.1f}%)\n"

            result += "\nTypical Examples (Top 5):\n"
            result += "-" * 20 + "\n"
            for warning in warnings[:5]:
                result += f"- {truncate_string(warning, 100)}\n"

            return result

        except Exception as e:
            logger.error(f"Analyze warnings failed: {e}")
            raise LogAnalysisError(f"Failed to analyze warnings in '{log_file}': {str(e)}")

    def _generate_report(self, log_file: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """Generate comprehensive log report"""
        try:
            if not self._validate_log_file(log_file):
                return f"Cannot access log file '{log_file}'."

            analysis = self._analyze_log_file(log_file, time_range)
            errors = self._analyze_errors(log_file, time_range)
            warnings = self._analyze_warnings(log_file, time_range)

            # Extract metrics using regex for consistency
            error_count = '0'
            match_err = re.search(r'Total Errors: (\d+)', errors)
            if match_err:
                error_count = match_err.group(1)

            warning_count = '0'
            match_warn = re.search(r'Total Warnings: (\d+)', warnings)
            if match_warn:
                warning_count = match_warn.group(1)

            summary = f"Comprehensive Log Report - '{log_file}'\n"
            summary += "=" * 60 + "\n"
            summary += f"Key Metrics: Errors={error_count}, Warnings={warning_count}\n\n"
            summary += "Detailed Analysis:\n"
            summary += "-" * 20 + "\n"
            summary += "1. Basic Analysis:\n"
            summary += analysis.split('Last 10 Log Lines:')[0]
            summary += "\n2. Error Analysis:\n"
            summary += errors.split('Typical Examples')[0]
            summary += "\n3. Warning Analysis:\n"
            summary += warnings.split('Typical Examples')[0]

            return truncate_string(summary, 3000)

        except Exception as e:
            logger.error(f"Generate report failed: {e}")
            raise LogAnalysisError(f"Failed to generate report for '{log_file}': {str(e)}")

    def _search_log(self, log_file: str, keyword: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """Search logs for keyword"""
        try:
            if not self._validate_log_file(log_file):
                return f"Cannot access log file '{log_file}'."

            content = self._read_log_file(log_file)

            filtered_content = content
            time_filter_info = ""
            if time_range:
                filtered_content = self._filter_logs_by_time(content, time_range)
                time_filter_info = f"(Time Range: {time_range[0]} to {time_range[1]})"

            lines = filtered_content.splitlines()
            matches = [line for line in lines if keyword.lower() in line.lower()]

            if not matches:
                return f"No entries found containing '{keyword}' in '{log_file}' {time_filter_info}."

            result = f"Found {len(matches)} matches for '{keyword}' in '{log_file}' {time_filter_info}:\n"
            result += "-" * 50 + "\n"

            for line in matches[:10]:
                log_time = self._parse_log_time(line)
                time_str = log_time.strftime("%Y-%m-%d %H:%M:%S") if log_time else "Unknown Time"
                result += f"[{time_str}] {truncate_string(line, 120)}\n"

            if len(matches) > 10:
                result += f"\n... and {len(matches) - 10} more matches"

            return result

        except Exception as e:
            logger.error(f"Search log failed: {e}")
            raise LogAnalysisError(f"Failed to search '{keyword}' in '{log_file}': {str(e)}")

    def _search_all_logs(self, keyword: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """Search keyword across all common system logs"""
        common_logs = [
            '/var/log/nginx/error.log', '/var/log/nginx/access.log',
            '/var/log/apache2/error.log', '/var/log/apache2/access.log',
            '/var/log/mysql/error.log', '/var/log/syslog',
            '/var/log/auth.log', '/var/log/kern.log'
        ]

        result = f"Global Search Results for '{keyword}':\n"
        result += "-" * 50 + "\n"

        for log_file in common_logs:
            try:
                if not self._validate_log_file(log_file):
                    continue

                content = self._read_log_file(log_file)
                filtered_content = content
                if time_range:
                    filtered_content = self._filter_logs_by_time(content, time_range)

                matches = [line for line in filtered_content.splitlines() if keyword.lower() in line.lower()]
                if matches:
                    result += f"\nFile: {log_file} ({len(matches)} matches):\n"
                    result += "-" * 30 + "\n"
                    for line in matches[:3]:
                        result += f"- {truncate_string(line, 100)}\n"
            except Exception as e:
                logger.warning(f"Failed to search {log_file}: {e}")
                result += f"\n⚠️  Error searching {log_file}: {str(e)}\n"

        return result

    def _get_log_statistics(self, log_file: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """Get log statistics"""
        try:
            if not self._validate_log_file(log_file):
                return f"Cannot access log file '{log_file}'."

            content = self._read_log_file(log_file)
            filtered_content = content
            time_filter_info = ""
            if time_range:
                filtered_content = self._filter_logs_by_time(content, time_range)
                time_filter_info = f"(Time Range: {time_range[0]} to {time_range[1]})"
                if not filtered_content:
                    return f"No log entries found in the specified time range {time_filter_info}"

            lines = filtered_content.splitlines()
            level_stats = parse_log_level(filtered_content)
            error_count = len(extract_error_patterns(filtered_content))

            time_buckets = {}
            for line in lines:
                log_time = self._parse_log_time(line)
                if log_time:
                    bucket_key = log_time.strftime("%Y-%m-%d %H:00")
                    time_buckets[bucket_key] = time_buckets.get(bucket_key, 0) + 1

            result = f"Log Statistics - '{log_file}' {time_filter_info}:\n"
            result += "=" * 50 + "\n"
            result += f"Total Lines: {len(lines)}\n"
            result += f"Total Errors: {error_count}\n"
            min_time = min(time_buckets.keys()) if time_buckets else 'Unknown'
            max_time = max(time_buckets.keys()) if time_buckets else 'Unknown'
            result += f"Time Coverage: {min_time} to {max_time}\n\n"

            result += "Level Distribution:\n"
            result += "-" * 20 + "\n"
            total = sum(level_stats.values())
            for level, count in level_stats.items():
                percentage = (count / total * 100) if total > 0 else 0
                result += f"{level}: {count} ({percentage:.1f}%)\n"

            if time_buckets:
                result += "\nTraffic Distribution (Hourly):\n"
                result += "-" * 30 + "\n"
                sorted_buckets = sorted(time_buckets.items())[-8:]
                for bucket, count in sorted_buckets:
                    result += f"{bucket}: {count} lines\n"

            return result

        except Exception as e:
            logger.error(f"Get statistics failed: {e}")
            raise LogAnalysisError(f"Failed to get statistics for '{log_file}': {str(e)}")

    # Helper methods
    def _validate_log_file(self, log_file: str) -> bool:
        """Validate if log file exists and size is within limits"""
        try:
            path = Path(log_file)
            if not path.exists() or not path.is_file():
                return False

            config = LogAnalyzerConfig()
            max_size = self._parse_size(config.max_file_size)
            if path.stat().st_size > max_size:
                return False

            return True
        except Exception:
            return False

    def _read_log_file(self, log_file: str) -> str:
        """Read log file content"""
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Read log file failed: {e}")
            raise FileOperationError(f"Cannot read log file '{log_file}': {str(e)}")

    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes"""
        size_str = size_str.upper().strip()

        if size_str.endswith('KB'):
            return int(float(size_str[:-2]) * 1024)
        elif size_str.endswith('MB'):
            return int(float(size_str[:-2]) * 1024 * 1024)
        elif size_str.endswith('GB'):
            return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
        else:
            return int(size_str)

    def _get_file_size(self, log_file: str) -> str:
        """Get human-readable file size"""
        try:
            size = Path(log_file).stat().st_size
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f}{unit}"
                size /= 1024.0
            return f"{size:.1f}TB"
        except Exception:
            return "Unknown"

    def _find_error_line(self, content: str, error_pattern: str) -> Optional[str]:
        """Find full log line containing the error pattern"""
        for line in content.splitlines():
            if error_pattern in line:
                return line
        return None