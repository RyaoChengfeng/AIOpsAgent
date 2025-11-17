"""
自定义异常模块
定义项目中使用的自定义异常类
"""


class AIOpsAgentException(Exception):
    """AI Agent基础异常类"""
    
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(AIOpsAgentException):
    """配置错误异常"""
    
    def __init__(self, message: str):
        super().__init__(message, "CONFIG_ERROR")


class DockerOperationError(AIOpsAgentException):
    """Docker操作错误异常"""
    
    def __init__(self, message: str, container_id: str = None):
        super().__init__(message, "DOCKER_ERROR")
        self.container_id = container_id


class SystemMonitorError(AIOpsAgentException):
    """系统监控错误异常"""
    
    def __init__(self, message: str):
        super().__init__(message, "MONITOR_ERROR")


class FileOperationError(AIOpsAgentException):
    """文件操作错误异常"""
    
    def __init__(self, message: str, file_path: str = None):
        super().__init__(message, "FILE_ERROR")
        self.file_path = file_path


class LogAnalysisError(AIOpsAgentException):
    """日志分析错误异常"""
    
    def __init__(self, message: str, log_file: str = None):
        super().__init__(message, "LOG_ERROR")
        self.log_file = log_file


class PerformanceAnalysisError(AIOpsAgentException):
    """性能分析错误异常"""
    
    def __init__(self, message: str):
        super().__init__(message, "PERF_ERROR")


class ServiceCheckError(AIOpsAgentException):
    """服务检查错误异常"""
    
    def __init__(self, message: str, service_name: str = None):
        super().__init__(message, "SERVICE_ERROR")
        self.service_name = service_name


class AIAgentError(AIOpsAgentException):
    """AI Agent错误异常"""
    
    def __init__(self, message: str):
        super().__init__(message, "AGENT_ERROR")


class CommandExecutionError(AIOpsAgentException):
    """命令执行错误异常"""
    
    def __init__(self, message: str, command: str = None, return_code: int = None):
        super().__init__(message, "CMD_ERROR")
        self.command = command
        self.return_code = return_code


class ValidationError(AIOpsAgentException):
    """验证错误异常"""
    
    def __init__(self, message: str, field: str = None):
        super().__init__(message, "VALIDATION_ERROR")
        self.field = field


class TimeoutError(AIOpsAgentException):
    """超时错误异常"""
    
    def __init__(self, message: str, timeout_seconds: int = None):
        super().__init__(message, "TIMEOUT_ERROR")
        self.timeout_seconds = timeout_seconds


class PermissionError(AIOpsAgentException):
    """权限错误异常"""
    
    def __init__(self, message: str, resource: str = None):
        super().__init__(message, "PERMISSION_ERROR")
        self.resource = resource


class NetworkError(AIOpsAgentException):
    """网络错误异常"""
    
    def __init__(self, message: str, host: str = None, port: int = None):
        super().__init__(message, "NETWORK_ERROR")
        self.host = host
        self.port = port


class APIError(AIOpsAgentException):
    """API错误异常"""
    
    def __init__(self, message: str, status_code: int = None, response: str = None):
        super().__init__(message, "API_ERROR")
        self.status_code = status_code
        self.response = response
