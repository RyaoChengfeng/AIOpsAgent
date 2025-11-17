"""
日志工具模块
提供统一的日志记录功能
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional
from config.settings import get_config


def setup_logger(
    name: str = "AIOpsAgent",
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径
        level: 日志级别
        format_string: 日志格式字符串
        
    Returns:
        配置好的日志记录器
    """
    # 获取配置
    if not log_file:
        log_file = get_config('logging.file', 'logs/agent.log')
    if not format_string:
        format_string = get_config(
            'logging.format',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # 创建日志目录
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 创建格式化器
    formatter = logging.Formatter(format_string)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（带轮转）
    max_size = get_config('logging.max_size', '10MB')
    backup_count = get_config('logging.backup_count', 5)
    
    # 解析文件大小
    size_bytes = _parse_size(max_size)
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=size_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def _parse_size(size_str: str) -> int:
    """
    解析文件大小字符串
    
    Args:
        size_str: 大小字符串，如 '10MB', '1GB'
        
    Returns:
        字节数
    """
    size_str = size_str.upper().strip()
    
    if size_str.endswith('KB'):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    else:
        # 默认为字节
        return int(size_str)


def get_logger(name: str = "AIOpsAgent") -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # 如果没有处理器，则设置默认配置
        level = get_config('logging.level', 'INFO')
        return setup_logger(name, level=level)
    return logger


class LoggerMixin:
    """日志记录器混入类"""
    
    @property
    def logger(self) -> logging.Logger:
        """获取日志记录器"""
        if not hasattr(self, '_logger'):
            class_name = self.__class__.__name__
            self._logger = get_logger(f"AIOpsAgent.{class_name}")
        return self._logger


# 创建默认日志记录器
default_logger = setup_logger()


def log_function_call(func):
    """
    装饰器：记录函数调用
    
    Args:
        func: 被装饰的函数
        
    Returns:
        装饰后的函数
    """
    def wrapper(*args, **kwargs):
        logger = get_logger()
        func_name = func.__name__
        logger.debug(f"调用函数: {func_name}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"函数 {func_name} 执行成功")
            return result
        except Exception as e:
            logger.error(f"函数 {func_name} 执行失败: {e}")
            raise
    
    return wrapper


def log_execution_time(func):
    """
    装饰器：记录函数执行时间
    
    Args:
        func: 被装饰的函数
        
    Returns:
        装饰后的函数
    """
    import time
    
    def wrapper(*args, **kwargs):
        logger = get_logger()
        func_name = func.__name__
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"函数 {func_name} 执行时间: {execution_time:.2f}秒")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"函数 {func_name} 执行失败 (耗时 {execution_time:.2f}秒): {e}")
            raise
    
    return wrapper
