"""
Utility Functions Module
Provides general-purpose helper functions
"""

import os
import re
import json
import subprocess
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import psutil
from src.utils.logger import get_logger

logger = get_logger(__name__)


def format_bytes(bytes_value: int) -> str:
    """
    Format a byte value into a human-readable string.

    Args:
        bytes_value: Byte count

    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"


def format_percentage(value: float, total: float) -> str:
    """
    Format percentage.

    Args:
        value: Current value
        total: Total value

    Returns:
        Percentage string
    """
    if total == 0:
        return "0.0%"
    percentage = (value / total) * 100
    return f"{percentage:.1f}%"


def format_duration(seconds: float) -> str:
    """
    Format a duration value.

    Args:
        seconds: Number of seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


def run_command(
    command: Union[str, List[str]],
    timeout: int = 30,
    capture_output: bool = True,
    shell: bool = True
) -> Tuple[int, str, str]:
    """
    Execute a system command.

    Args:
        command: Command to execute
        timeout: Timeout in seconds
        capture_output: Whether to capture output
        shell: Execute in shell mode

    Returns:
        (return_code, stdout, stderr)
    """
    try:
        logger.debug(f"Executing command: {command}")

        result = subprocess.run(
            command,
            timeout=timeout,
            capture_output=capture_output,
            text=True,
            shell=shell
        )

        logger.debug(f"Command completed, return code: {result.returncode}")
        return result.returncode, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out: {command}")
        return -1, "", "Command execution timed out"
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        return -1, "", str(e)


def is_port_open(host: str, port: int, timeout: int = 5) -> bool:
    """
    Check if a port is open.

    Args:
        host: Host address
        port: Port number
        timeout: Timeout

    Returns:
        Whether the port is open
    """
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            return result == 0
    except Exception:
        return False


def get_process_by_name(name: str) -> List[Dict[str, Any]]:
    """
    Get process info by name.

    Args:
        name: Process name

    Returns:
        List of process information
    """
    processes = []

    try:
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            if name.lower() in proc.info['name'].lower():
                processes.append(proc.info)
    except Exception as e:
        logger.error(f"Failed to get process info: {e}")

    return processes


def get_process_by_port(port: int) -> Optional[Dict[str, Any]]:
    """
    Get process info by port number.

    Args:
        port: Port number

    Returns:
        Process info or None
    """
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                connections = proc.connections()
                for conn in connections:
                    if conn.laddr.port == port:
                        return {
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'port': port
                        }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        logger.error(f"Failed to get process by port: {e}")

    return None


def parse_log_level(log_content: str) -> Dict[str, int]:
    """
    Parse log content and count log levels.

    Args:
        log_content: Log content

    Returns:
        Count per log level
    """
    levels = {
        'DEBUG': 0,
        'INFO': 0,
        'WARNING': 0,
        'ERROR': 0,
        'CRITICAL': 0
    }

    for line in log_content.split('\n'):
        for level in levels.keys():
            if level in line.upper():
                levels[level] += 1
                break

    return levels


def extract_error_patterns(log_content: str) -> List[str]:
    """
    Extract error patterns from log text.

    Args:
        log_content: Log content

    Returns:
        List of error patterns
    """
    error_patterns = []

    patterns = [
        r'Exception.*?:.*',
        r'Error.*?:.*',
        r'Failed.*',
        r'Timeout.*',
        r'Connection.*refused',
        r'Permission.*denied',
        r'No such file.*',
        r'Cannot.*'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, log_content, re.IGNORECASE)
        error_patterns.extend(matches)

    return list(set(error_patterns))


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON.

    Args:
        json_str: JSON string
        default: Default value if parsing fails

    Returns:
        Parsed result or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """
    Safely serialize object to JSON.

    Args:
        obj: Object to serialize
        default: Default return value if serialization fails

    Returns:
        JSON string or default
    """
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return default


def validate_file_path(file_path: str, must_exist: bool = True) -> bool:
    """
    Validate a file path.

    Args:
        file_path: File path
        must_exist: Whether file must exist

    Returns:
        Whether the path is valid
    """
    try:
        path = Path(file_path)

        if must_exist:
            return path.exists() and path.is_file()
        else:
            return path.parent.exists()
    except Exception:
        return False


def create_backup_filename(original_path: str) -> str:
    """
    Create a backup file name.

    Args:
        original_path: Original file path

    Returns:
        Backup file path
    """
    path = Path(original_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{path.stem}_{timestamp}{path.suffix}"
    return str(path.parent / backup_name)


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get basic file information.

    Args:
        file_path: File path

    Returns:
        Dictionary containing file info
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return {}

        stat = path.stat()
        return {
            'name': path.name,
            'size': stat.st_size,
            'size_formatted': format_bytes(stat.st_size),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'is_file': path.is_file(),
            'is_dir': path.is_dir(),
            'extension': path.suffix,
            'permissions': oct(stat.st_mode)[-3:]
        }
    except Exception as e:
        logger.error(f"Failed to get file info: {e}")
        return {}


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    Retry decorator.

    Args:
        max_retries: Max retry count
        delay: Delay between retries (seconds)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Function {func.__name__} failed on attempt {attempt + 1}: {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"Function {func.__name__} still failed after {max_retries} retries")

            raise last_exception

        return wrapper
    return decorator


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string.

    Args:
        text: Original text
        max_length: Max length allowed
        suffix: Truncation suffix

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def get_system_info() -> Dict[str, Any]:
    """
    Get system basic information.

    Returns:
        Dictionary containing system information
    """
    try:
        import platform

        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'hostname': platform.node(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'boot_time': datetime.fromtimestamp(psutil.boot_time())
        }
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return {}
