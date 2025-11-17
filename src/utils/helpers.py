"""
辅助函数模块
提供通用的工具函数
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
    格式化字节数为人类可读的格式
    
    Args:
        bytes_value: 字节数
        
    Returns:
        格式化后的字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"


def format_percentage(value: float, total: float) -> str:
    """
    格式化百分比
    
    Args:
        value: 当前值
        total: 总值
        
    Returns:
        百分比字符串
    """
    if total == 0:
        return "0.0%"
    percentage = (value / total) * 100
    return f"{percentage:.1f}%"


def format_duration(seconds: float) -> str:
    """
    格式化时间间隔
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化后的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}小时"
    else:
        days = seconds / 86400
        return f"{days:.1f}天"


def run_command(
    command: Union[str, List[str]],
    timeout: int = 30,
    capture_output: bool = True,
    shell: bool = True
) -> Tuple[int, str, str]:
    """
    执行系统命令
    
    Args:
        command: 要执行的命令
        timeout: 超时时间（秒）
        capture_output: 是否捕获输出
        shell: 是否使用shell执行
        
    Returns:
        (返回码, 标准输出, 标准错误)
    """
    try:
        logger.debug(f"执行命令: {command}")
        
        result = subprocess.run(
            command,
            timeout=timeout,
            capture_output=capture_output,
            text=True,
            shell=shell
        )
        
        logger.debug(f"命令执行完成，返回码: {result.returncode}")
        return result.returncode, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        logger.error(f"命令执行超时: {command}")
        return -1, "", "命令执行超时"
    except Exception as e:
        logger.error(f"命令执行失败: {e}")
        return -1, "", str(e)


def is_port_open(host: str, port: int, timeout: int = 5) -> bool:
    """
    检查端口是否开放
    
    Args:
        host: 主机地址
        port: 端口号
        timeout: 超时时间
        
    Returns:
        端口是否开放
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
    根据进程名获取进程信息
    
    Args:
        name: 进程名
        
    Returns:
        进程信息列表
    """
    processes = []
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            if name.lower() in proc.info['name'].lower():
                processes.append(proc.info)
    except Exception as e:
        logger.error(f"获取进程信息失败: {e}")
    
    return processes


def get_process_by_port(port: int) -> Optional[Dict[str, Any]]:
    """
    根据端口号获取进程信息
    
    Args:
        port: 端口号
        
    Returns:
        进程信息或None
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
        logger.error(f"根据端口获取进程信息失败: {e}")
    
    return None


def parse_log_level(log_content: str) -> Dict[str, int]:
    """
    解析日志内容，统计各级别日志数量
    
    Args:
        log_content: 日志内容
        
    Returns:
        各级别日志数量统计
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
    从日志中提取错误模式
    
    Args:
        log_content: 日志内容
        
    Returns:
        错误模式列表
    """
    error_patterns = []
    
    # 常见错误模式
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
    
    return list(set(error_patterns))  # 去重


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    安全的JSON解析
    
    Args:
        json_str: JSON字符串
        default: 解析失败时的默认值
        
    Returns:
        解析结果或默认值
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """
    安全的JSON序列化
    
    Args:
        obj: 要序列化的对象
        default: 序列化失败时的默认值
        
    Returns:
        JSON字符串或默认值
    """
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return default


def validate_file_path(file_path: str, must_exist: bool = True) -> bool:
    """
    验证文件路径
    
    Args:
        file_path: 文件路径
        must_exist: 文件是否必须存在
        
    Returns:
        路径是否有效
    """
    try:
        path = Path(file_path)
        
        if must_exist:
            return path.exists() and path.is_file()
        else:
            # 检查父目录是否存在
            return path.parent.exists()
    except Exception:
        return False


def create_backup_filename(original_path: str) -> str:
    """
    创建备份文件名
    
    Args:
        original_path: 原始文件路径
        
    Returns:
        备份文件路径
    """
    path = Path(original_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{path.stem}_{timestamp}{path.suffix}"
    return str(path.parent / backup_name)


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    获取文件信息
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件信息字典
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
        logger.error(f"获取文件信息失败: {e}")
        return {}


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 重试间隔（秒）
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
                        logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"函数 {func.__name__} 重试 {max_retries} 次后仍然失败")
            
            raise last_exception
        
        return wrapper
    return decorator


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断字符串
    
    Args:
        text: 原始字符串
        max_length: 最大长度
        suffix: 截断后缀
        
    Returns:
        截断后的字符串
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def get_system_info() -> Dict[str, Any]:
    """
    获取系统基本信息
    
    Returns:
        系统信息字典
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
        logger.error(f"获取系统信息失败: {e}")
        return {}
