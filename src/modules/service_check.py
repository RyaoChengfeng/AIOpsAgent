"""
服务检查模块
提供系统服务状态检查和管理的工具函数和LangChain工具
"""

import subprocess
import psutil
import socket
from typing import Dict, List, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from config.settings import get_config
from src.utils.logger import get_logger
from src.utils.helpers import run_command, is_port_open, get_process_by_name, get_process_by_port
from src.utils.exceptions import ServiceCheckError, CommandExecutionError

logger = get_logger(__name__)


class ServiceConfig(BaseModel):
    """服务检查配置模型"""
    timeout: int = Field(default_factory=lambda: get_config('service_check.timeout', 10))
    retry_count: int = Field(default_factory=lambda: get_config('service_check.retry_count', 3))
    retry_delay: int = Field(default_factory=lambda: get_config('service_check.retry_delay', 5))


class ServiceCheckerTool(BaseTool):
    """服务检查LangChain工具"""
    
    name: str = "service_checker"
    description: str = (
        "用于检查和管理系统服务的工具。支持服务状态查询、自动重启失败服务、"
        "端口服务检查、进程服务监控等操作。"
        "输入应为具体的服务检查请求，如'检查nginx服务状态'、'重启mysql服务'、"
        "'检查端口80的服务'或'列出所有运行的服务'"
    )
    args_schema: Optional[BaseModel] = None
    
    def _run(self, check_request: str) -> str:
        """
        执行服务检查操作
        
        Args:
            check_request: 服务检查请求描述
            
        Returns:
            检查结果
        """
        try:
            request_lower = check_request.lower()
            
            if "检查" in request_lower or "check" in request_lower:
                service_name = self._extract_service_name(check_request)
                if service_name:
                    return self._check_service_status(service_name)
                else:
                    return self._list_all_services()
            elif "重启" in request_lower or "restart" in request_lower:
                service_name = self._extract_service_name(check_request)
                if service_name:
                    return self._restart_service(service_name)
                else:
                    return "请指定要重启的服务名称。"
            elif "启动" in request_lower or "start" in request_lower:
                service_name = self._extract_service_name(check_request)
                if service_name:
                    return self._start_service(service_name)
                else:
                    return "请指定要启动的服务名称。"
            elif "停止" in request_lower or "stop" in request_lower:
                service_name = self._extract_service_name(check_request)
                if service_name:
                    return self._stop_service(service_name)
                else:
                    return "请指定要停止的服务名称。"
            elif "端口" in request_lower or "port" in request_lower:
                port = self._extract_port(check_request)
                if port:
                    return self._check_port_service(port)
                else:
                    return "请指定端口号。"
            else:
                return (
                    "支持的服务操作:\\n"
                    "- 检查服务状态 (指定服务名)\\n"
                    "- 重启/启动/停止服务 (指定服务名)\\n"
                    "- 检查端口服务 (指定端口号)\\n"
                    "- 列出所有运行服务\\n"
                    "示例: '检查nginx服务状态' 或 '重启apache2服务'"
                )
                
        except Exception as e:
            logger.error(f"服务检查失败: {e}")
            raise ServiceCheckError(f"服务检查执行失败: {str(e)}")
    
    def _extract_service_name(self, request: str) -> Optional[str]:
        """从请求中提取服务名称"""
        # 常见服务名
        common_services = ['nginx', 'apache2', 'mysql', 'postgresql', 'redis', 'mongodb', 
                          'docker', 'systemd', 'sshd', 'httpd']
        
        for service in common_services:
            if service in request.lower():
                return service
        
        # 提取最后一个词作为服务名
        words = request.split()
        if len(words) > 1:
            return words[-1]
        
        return None
    
    def _extract_port(self, request: str) -> Optional[int]:
        """从请求中提取端口号"""
        import re
        port_match = re.search(r'端口\s*(\d+)', request, re.IGNORECASE)
        if port_match:
            return int(port_match.group(1))
        return None
    
    def _check_service_status(self, service_name: str) -> str:
        """检查服务状态"""
        config = ServiceConfig()
        
        try:
            # 首先尝试systemctl (Linux)
            result = run_command(f"systemctl is-active {service_name}", timeout=config.timeout)
            if result[0] == 0:
                status = result[1].strip()
                if status == "active":
                    return self._get_detailed_service_info(service_name, "running")
                else:
                    return self._get_detailed_service_info(service_name, status)
            elif "not-found" in result[2].lower():
                # 服务不存在，尝试进程检查
                return self._check_by_process(service_name)
            else:
                return f"❌ 检查服务 '{service_name}' 失败: {result[2]}"
                
        except Exception as e:
            logger.error(f"检查服务状态失败: {e}")
            raise ServiceCheckError(f"无法检查服务 '{service_name}' 状态: {str(e)}")
    
    def _get_detailed_service_info(self, service_name: str, status: str) -> str:
        """获取详细的服务信息"""
        try:
            # 获取服务详细信息
            result = run_command(f"systemctl status {service_name} --no-pager", timeout=10)
            
            if result[0] == 0:
                output = result[1]
                # 提取关键信息
                lines = output.split('\n')
                info_lines = []
                
                for line in lines[:10]:  # 前10行通常包含重要信息
                    if any(keyword in line.lower() for keyword in ['active', 'loaded', 'main pid', 'since']):
                        info_lines.append(line.strip())
                
                detailed_info = '\n'.join(info_lines)
                
                result_str = f"服务 '{service_name}' 状态: {status}\\n"
                result_str += "=" * 40 + "\n"
                result_str += detailed_info
                
                if status != "running":
                    result_str += f"\n💡 建议: 服务 '{service_name}' 未运行，考虑使用 '重启{service_name}服务' 命令。"
                
                return result_str
            else:
                return f"无法获取服务 '{service_name}' 详细信息: {result[2]}"
                
        except Exception as e:
            logger.error(f"获取服务详细信息失败: {e}")
            return f"服务 '{service_name}' 状态: {status} (详细信息获取失败)"
    
    def _check_by_process(self, service_name: str) -> str:
        """通过进程检查服务状态"""
        try:
            # 查找相关进程
            processes = get_process_by_name(service_name)
            
            if processes:
                result = f"服务 '{service_name}' 相关进程 (运行中):\\n"
                result += "-" * 40 + "\n"
                
                for proc in processes[:5]:  # 显示前5个进程
                    result += f"PID: {proc['pid']}, 名称: {proc['name']}\\n"
                    result += f"CPU: {proc.get('cpu_percent', 0):.1f}%, 内存: {proc.get('memory_percent', 0):.1f}%\\n\n"
                
                return result
            else:
                return f"❌ 未找到服务 '{service_name}' 相关进程。服务可能未运行或服务名不正确。"
                
        except Exception as e:
            logger.error(f"进程检查失败: {e}")
            return f"检查服务 '{service_name}' 进程失败: {str(e)}"
    
    def _list_all_services(self) -> str:
        """列出所有服务"""
        try:
            # 列出运行中的服务
            result = run_command("systemctl list-units --type=service --state=running --no-pager", timeout=15)
            
            if result[0] == 0:
                output = result[1]
                lines = output.split('\n')
                
                running_services = [line.split()[0] for line in lines[1:] if line.strip() and not line.startswith('UNIT')]
                
                result_str = f"运行中的系统服务 ({len(running_services)} 个):\\n"
                result_str += "-" * 40 + "\n"
                
                # 显示前20个
                for service in running_services[:20]:
                    service_name = service.split('.')[0]  # 移除.service后缀
                    result_str += f"- {service_name}\n"
                
                if len(running_services) > 20:
                    result_str += f"\n... 还有 {len(running_services) - 20} 个运行服务"
                
                return result_str
            else:
                # Fallback到ps命令
                ps_result = run_command("ps aux --no-headers | wc -l", timeout=5)
                if ps_result[0] == 0:
                    process_count = int(ps_result[1].strip())
                    return f"系统当前运行进程数: {process_count}\n(无法获取systemd服务列表，使用ps命令统计)"
                else:
                    return "无法获取服务列表。"
                    
        except Exception as e:
            logger.error(f"列出服务失败: {e}")
            raise ServiceCheckError(f"无法列出服务: {str(e)}")
    
    def _restart_service(self, service_name: str) -> str:
        """重启服务"""
        config = ServiceConfig()
        
        # 安全确认 - 在实际使用中应该有用户确认
        confirmation_msg = (
            f"⚠️  警告: 即将重启服务 '{service_name}'。这可能会中断正在使用该服务的连接。\n"
            "请确认是否继续？(在生产环境中需要人工确认)\n\n"
        )
        
        try:
            # 尝试重启
            result = run_command(f"systemctl restart {service_name}", timeout=config.timeout)
            
            if result[0] == 0:
                # 验证重启成功
                time.sleep(2)  # 等待服务重启
                status_result = self._check_service_status(service_name)
                
                return confirmation_msg + f"✅ 服务 '{service_name}' 重启成功！\n\n{status_result}"
            else:
                return confirmation_msg + f"❌ 重启服务 '{service_name}' 失败: {result[2]}\n请检查服务配置和依赖。"
                
        except Exception as e:
            logger.error(f"重启服务失败: {e}")
            return confirmation_msg + f"❌ 重启服务 '{service_name}' 失败: {str(e)}"
    
    def _start_service(self, service_name: str) -> str:
        """启动服务"""
        config = ServiceConfig()
        
        try:
            result = run_command(f"systemctl start {service_name}", timeout=config.timeout)
            
            if result[0] == 0:
                # 验证启动成功
                time.sleep(1)
                status_result = self._check_service_status(service_name)
                
                return f"✅ 服务 '{service_name}' 启动成功！\n\n{status_result}"
            else:
                return f"❌ 启动服务 '{service_name}' 失败: {result[2]}\n请检查服务配置。"
                
        except Exception as e:
            logger.error(f"启动服务失败: {e}")
            raise ServiceCheckError(f"无法启动服务 '{service_name}': {str(e)}")
    
    def _stop_service(self, service_name: str) -> str:
        """停止服务"""
        config = ServiceConfig()
        
        warning_msg = (
            f"⚠️  警告: 即将停止服务 '{service_name}'。这会中断所有依赖该服务的连接。\n"
            "请确认是否继续？\n\n"
        )
        
        try:
            result = run_command(f"systemctl stop {service_name}", timeout=config.timeout)
            
            if result[0] == 0:
                return warning_msg + f"✅ 服务 '{service_name}' 已停止。"
            else:
                return warning_msg + f"❌ 停止服务 '{service_name}' 失败: {result[2]}"
                
        except Exception as e:
            logger.error(f"停止服务失败: {e}")
            raise ServiceCheckError(f"无法停止服务 '{service_name}': {str(e)}")
    
    def _check_port_service(self, port: int) -> str:
        """检查端口服务"""
        try:
            if not 1 <= port <= 65535:
                return "端口号必须在1-65535范围内。"
            
            # 检查端口是否开放
            is_open = is_port_open('localhost', port)
            
            # 查找占用端口的进程
            process = get_process_by_port(port)
            
            result = f"端口 {port} 服务检查:\\n"
            result += "=" * 30 + "\n"
            result += f"端口状态: {'🟢 开放' if is_open else '🔴 关闭'}\n"
            
            if process:
                result += f"\n占用进程:\n"
                result += f"  名称: {process['name']}\n"
                result += f"  PID: {process['pid']}\n"
                
                try:
                    proc = psutil.Process(process['pid'])
                    result += f"  命令行: {' '.join(proc.cmdline())}\n"
                    result += f"  CPU使用: {proc.cpu_percent():.1f}%\n"
                    result += f"  内存使用: {format_bytes(proc.memory_info().rss)}\n"
                    result += f"  启动时间: {datetime.fromtimestamp(proc.create_time()).strftime('%Y-%m-%d %H:%M:%S')}\n"
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    result += "  (无法获取详细信息 - 权限不足)\n"
                
                # 常见端口服务映射
                common_ports = {
                    22: "SSH",
                    80: "HTTP (Apache/Nginx)",
                    443: "HTTPS",
                    3306: "MySQL",
                    5432: "PostgreSQL",
                    6379: "Redis",
                    27017: "MongoDB"
                }
                
                service_name = common_ports.get(port, "未知服务")
                result += f"\n可能的服务: {service_name}"
                
            else:
                result += "\n当前没有进程占用该端口。\n"
                if is_open:
                    result += "端口开放但无进程占用，可能存在安全风险。"
                else:
                    result += "端口关闭，服务未运行。"
            
            return result
            
        except Exception as e:
            logger.error(f"端口服务检查失败: {e}")
            raise ServiceCheckError(f"无法检查端口 {port} 服务: {str(e)}")


if __name__ == "__main__":
    # 测试服务检查工具
    try:
        tool = ServiceCheckerTool()
        print("测试服务检查工具:")
        print(tool._run("列出所有运行的服务"))
    except Exception as e:
        print(f"测试失败: {e}")
