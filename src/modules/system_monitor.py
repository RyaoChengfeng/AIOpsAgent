"""
系统监控模块
提供系统资源监控的工具函数和LangChain工具
"""

import psutil
import socket
from typing import Dict, List, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from config.settings import get_config
from src.utils.logger import get_logger
from src.utils.helpers import format_bytes, format_percentage, get_process_by_name, get_process_by_port, get_system_info
from src.utils.exceptions import SystemMonitorError

logger = get_logger(__name__)


class MonitorConfig(BaseModel):
    """监控配置模型"""
    cpu_threshold: float = Field(default_factory=lambda: get_config('monitoring.cpu_threshold', 80.0))
    memory_threshold: float = Field(default_factory=lambda: get_config('monitoring.memory_threshold', 85.0))
    disk_threshold: float = Field(default_factory=lambda: get_config('monitoring.disk_threshold', 90.0))
    check_interval: int = Field(default_factory=lambda: get_config('monitoring.check_interval', 30))


class SystemMonitorTool(BaseTool):
    """系统监控LangChain工具"""
    
    name: str = "system_monitor"
    description: str = (
        "用于监控系统资源的工具。支持CPU、内存、磁盘、网络使用情况监控，"
        "端口占用检查，进程信息查询等操作。"
        "输入应为具体的监控请求，如'检查CPU和内存使用情况'、'显示占用端口的进程'或'检查磁盘空间'"
    )
    args_schema: Optional[BaseModel] = None
    
    def _run(self, query: str) -> str:
        """
        执行系统监控操作
        
        Args:
            query: 监控查询描述
            
        Returns:
            监控结果
        """
        try:
            query_lower = query.lower()
            
            if "cpu" in query_lower or "处理器" in query_lower:
                return self._get_cpu_info()
            elif "内存" in query_lower or "memory" in query_lower:
                return self._get_memory_info()
            elif "磁盘" in query_lower or "disk" in query_lower:
                return self._get_disk_info()
            elif "网络" in query_lower or "network" in query_lower:
                return self._get_network_info()
            elif "端口" in query_lower or "port" in query_lower:
                port = self._extract_port(query)
                if port:
                    return self._get_port_info(port)
                else:
                    return self._get_port_usage()
            elif "进程" in query_lower or "process" in query_lower:
                process_name = self._extract_process_name(query)
                if process_name:
                    return self._get_process_info(process_name)
                else:
                    return self._get_top_processes()
            elif "系统信息" in query_lower or "system info" in query_lower:
                return self._get_system_overview()
            else:
                return (
                    "支持的监控操作:\\n"
                    "- CPU使用情况\\n"
                    "- 内存使用情况\\n"
                    "- 磁盘空间使用\\n"
                    "- 网络流量监控\\n"
                    "- 端口占用检查 (指定端口号)\\n"
                    "- 进程信息查询 (指定进程名)\\n"
                    "- 系统整体概览\\n"
                    "请提供更具体的监控请求。"
                )
                
        except Exception as e:
            logger.error(f"系统监控失败: {e}")
            raise SystemMonitorError(f"系统监控执行失败: {str(e)}")
    
    def _get_cpu_info(self) -> str:
        """获取CPU信息"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count(logical=True)
            cpu_times = psutil.cpu_times_percent(interval=1)
            
            config = MonitorConfig()
            alert = cpu_percent > config.cpu_threshold
            
            result = "系统CPU信息:\\n"
            result += f"CPU核心数: {cpu_count} (逻辑)\\n"
            result += f"当前使用率: {cpu_percent:.1f}%\\n"
            result += f"用户时间: {cpu_times.user:.1f}%\\n"
            result += f"系统时间: {cpu_times.system:.1f}%\\n"
            result += f"空闲时间: {cpu_times.idle:.1f}%\\n"
            
            if alert:
                result += f"⚠️  CPU使用率超过阈值 ({config.cpu_threshold}%)，建议检查高负载进程。"
            
            return result
        except Exception as e:
            logger.error(f"获取CPU信息失败: {e}")
            raise SystemMonitorError("无法获取CPU信息")
    
    def _get_memory_info(self) -> str:
        """获取内存信息"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            config = MonitorConfig()
            memory_alert = memory.percent > config.memory_threshold
            
            result = "系统内存信息:\\n"
            result += f"总内存: {format_bytes(memory.total)}\\n"
            result += f"已使用: {format_bytes(memory.used)} ({memory.percent:.1f}%)\\n"
            result += f"可用: {format_bytes(memory.available)} ({memory.available / memory.total * 100:.1f}%)\\n"
            result += f"缓存: {format_bytes(memory.cached)}\\n"
            result += f"交换空间使用: {format_bytes(swap.used)} / {format_bytes(swap.total)} ({swap.percent:.1f}%)\\n"
            
            if memory_alert:
                result += f"⚠️  内存使用率超过阈值 ({config.memory_threshold}%)，建议关闭不必要的进程。"
            
            return result
        except Exception as e:
            logger.error(f"获取内存信息失败: {e}")
            raise SystemMonitorError("无法获取内存信息")
    
    def _get_disk_info(self) -> str:
        """获取磁盘信息"""
        try:
            partitions = psutil.disk_partitions()
            result = "磁盘使用情况:\\n"
            result += "设备\\t\\t总空间\\t\\t已使用\\t\\t使用率\\t\\t挂载点\\n"
            result += "-" * 60 + "\\n"
            
            config = MonitorConfig()
            alerts = []
            
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    usage_percent = (usage.used / usage.total) * 100
                    
                    result += f"{partition.device}\\t{format_bytes(usage.total)}\\t{format_bytes(usage.used)}\\t{usage_percent:.1f}%\\t{partition.mountpoint}\\n"
                    
                    if usage_percent > config.disk_threshold:
                        alerts.append(f"{partition.mountpoint}: {usage_percent:.1f}%")
                except PermissionError:
                    # 跳过无权限访问的分区
                    continue
            
            if alerts:
                result += f"\\n⚠️  以下磁盘使用率超过阈值 ({config.disk_threshold}%):\\n"
                for alert in alerts:
                    result += f"  - {alert}\\n"
            
            return result
        except Exception as e:
            logger.error(f"获取磁盘信息失败: {e}")
            raise SystemMonitorError("无法获取磁盘信息")
    
    def _get_network_info(self) -> str:
        """获取网络信息"""
        try:
            net_io = psutil.net_io_counters()
            connections = psutil.net_connections(kind='inet')
            
            result = "网络信息:\\n"
            result += f"发送字节: {format_bytes(net_io.bytes_sent)}\\n"
            result += f"接收字节: {format_bytes(net_io.bytes_recv)}\\n"
            result += f"发送数据包: {net_io.packets_sent}\\n"
            result += f"接收数据包: {net_io.packets_recv}\\n"
            result += f"网络连接数: {len(connections)}\\n"
            
            # 显示前10个活跃连接
            active_connections = [conn for conn in connections if conn.status == 'ESTABLISHED']
            if active_connections:
                result += "\\n活跃TCP连接 (前10个):\\n"
                result += "本地地址\\t\\t远程地址\\t\\t状态\\n"
                result += "-" * 40 + "\\n"
                
                for conn in active_connections[:10]:
                    local = f"{conn.laddr.ip}:{conn.laddr.port}"
                    remote = f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "*:*"
                    result += f"{local}\\t{remote}\\t{conn.status}\\n"
            
            return result
        except Exception as e:
            logger.error(f"获取网络信息失败: {e}")
            raise SystemMonitorError("无法获取网络信息")
    
    def _get_port_info(self, port: int) -> str:
        """获取指定端口信息"""
        try:
            if not 1 <= port <= 65535:
                return "端口号必须在1-65535范围内。"
            
            # 检查端口是否开放
            is_open = self._is_port_open('localhost', port)
            
            # 查找占用端口的进程
            process = get_process_by_port(port)
            
            result = f"端口 {port} 信息:\\n"
            result += f"状态: {'开放' if is_open else '关闭'}\\n"
            
            if process:
                result += f"占用进程: {process['name']} (PID: {process['pid']})\\n"
                try:
                    proc = psutil.Process(process['pid'])
                    result += f"进程命令行: {proc.cmdline()}\\n"
                    result += f"进程CPU使用: {proc.cpu_percent():.1f}%\\n"
                    result += f"进程内存使用: {format_bytes(proc.memory_info().rss)}\\n"
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    result += "无法获取进程详细信息（权限不足）\\n"
            else:
                result += "当前没有进程占用该端口\\n"
            
            return result
        except Exception as e:
            logger.error(f"获取端口信息失败: {e}")
            raise SystemMonitorError(f"无法获取端口 {port} 信息")
    
    def _get_port_usage(self) -> str:
        """获取端口使用情况"""
        try:
            connections = psutil.net_connections(kind='tcp')
            listening_ports = {}
            
            for conn in connections:
                if conn.status == 'LISTEN':
                    port = conn.laddr.port
                    if port not in listening_ports:
                        listening_ports[port] = []
                    listening_ports[port].append(conn.pid)
            
            result = f"监听端口使用情况 (前10个):\\n"
            result += "端口\\t\\t占用进程PID\\n"
            result += "-" * 30 + "\\n"
            
            for port in sorted(listening_ports.keys())[:10]:
                pids = listening_ports[port]
                pid_str = ", ".join(map(str, pids[:3])) + ("..." if len(pids) > 3 else "")
                result += f"{port}\\t\\t{pid_str}\\n"
            
            if len(listening_ports) > 10:
                result += f"\\n... 还有 {len(listening_ports) - 10} 个端口在监听"
            
            return result
        except Exception as e:
            logger.error(f"获取端口使用情况失败: {e}")
            raise SystemMonitorError("无法获取端口使用情况")
    
    def _get_process_info(self, process_name: str) -> str:
        """获取指定进程信息"""
        try:
            processes = get_process_by_name(process_name)
            
            if not processes:
                return f"未找到名为 '{process_name}' 的进程。"
            
            result = f"进程 '{process_name}' 信息:\\n"
            result += "-" * 40 + "\\n"
            
            for proc in processes:
                result += f"PID: {proc['pid']}\\n"
                result += f"名称: {proc['name']}\\n"
                result += f"CPU使用率: {proc.get('cpu_percent', 0):.1f}%\\n"
                result += f"内存使用率: {proc.get('memory_percent', 0):.1f}%\\n"
                
                try:
                    full_proc = psutil.Process(proc['pid'])
                    result += f"命令行: {' '.join(full_proc.cmdline())}\\n"
                    result += f"启动时间: {datetime.fromtimestamp(full_proc.create_time()).strftime('%Y-%m-%d %H:%M:%S')}\\n"
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    result += "无法获取详细信息（权限不足）\\n"
                
                result += "\\n"
            
            return result
        except Exception as e:
            logger.error(f"获取进程信息失败: {e}")
            raise SystemMonitorError(f"无法获取进程 '{process_name}' 信息")
    
    def _get_top_processes(self, limit: int = 10) -> str:
        """获取占用资源最多的进程"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # 按CPU使用率排序
            cpu_sorted = sorted(processes, key=lambda x: x.get('cpu_percent', 0), reverse=True)
            # 按内存使用率排序
            memory_sorted = sorted(processes, key=lambda x: x.get('memory_percent', 0), reverse=True)
            
            result = "资源占用最高的进程 (Top 10):\\n"
            result += "\\nCPU占用最高:\\n"
            result += "PID\\t\\t名称\\t\\tCPU%\\t\\t内存%\\n"
            result += "-" * 40 + "\\n"
            
            for proc in cpu_sorted[:limit]:
                result += f"{proc['pid']}\\t\\t{proc['name']}\\t\\t{proc.get('cpu_percent', 0):.1f}%\\t\\t{proc.get('memory_percent', 0):.1f}%\\n"
            
            result += "\\n内存占用最高:\\n"
            result += "PID\\t\\t名称\\t\\tCPU%\\t\\t内存%\\n"
            result += "-" * 40 + "\\n"
            
            for proc in memory_sorted[:limit]:
                result += f"{proc['pid']}\\t\\t{proc['name']}\\t\\t{proc.get('cpu_percent', 0):.1f}%\\t\\t{proc.get('memory_percent', 0):.1f}%\\n"
            
            return result
        except Exception as e:
            logger.error(f"获取Top进程失败: {e}")
            raise SystemMonitorError("无法获取进程信息")
    
    def _get_system_overview(self) -> str:
        """获取系统整体概览"""
        try:
            system_info = get_system_info()
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent
            
            result = "系统整体概览:\\n"
            result += f"操作系统: {system_info.get('system', '未知')} {system_info.get('release', '未知')}\\n"
            result += f"主机名: {system_info.get('hostname', '未知')}\\n"
            result += f"CPU核心数: {system_info.get('cpu_count', '未知')}\\n"
            result += f"总内存: {format_bytes(system_info.get('memory_total', 0))}\\n"
            result += f"Python版本: {system_info.get('python_version', '未知')}\\n"
            result += f"启动时间: {system_info.get('boot_time', '未知').strftime('%Y-%m-%d %H:%M:%S') if system_info.get('boot_time') else '未知'}\\n"
            result += "\\n当前资源使用:\\n"
            result += f"CPU使用率: {cpu:.1f}%\\n"
            result += f"内存使用率: {memory:.1f}%\\n"
            result += f"根分区使用率: {disk:.1f}%\\n"
            
            return result
        except Exception as e:
            logger.error(f"获取系统概览失败: {e}")
            raise SystemMonitorError("无法获取系统概览")
    
    def _extract_port(self, query: str) -> Optional[int]:
        """从查询中提取端口号"""
        import re
        port_match = re.search(r'端口\s*(\d+)', query, re.IGNORECASE)
        if port_match:
            return int(port_match.group(1))
        return None
    
    def _extract_process_name(self, query: str) -> Optional[str]:
        """从查询中提取进程名"""
        # 简单提取最后一个词作为进程名
        words = query.split()
        if len(words) > 1:
            return words[-1]
        return None
    
    def _is_port_open(self, host: str, port: int, timeout: int = 3) -> bool:
        """检查端口是否开放"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                result = sock.connect_ex((host, port))
                return result == 0
        except Exception:
            return False


if __name__ == "__main__":
    # 测试系统监控工具
    try:
        tool = SystemMonitorTool()
        print("测试系统监控工具:")
        print(tool._run("检查CPU和内存使用情况"))
    except Exception as e:
        print(f"测试失败: {e}")
