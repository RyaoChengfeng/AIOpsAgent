"""
System monitoring module (AI-driven, multi-command)
Provides CPU, memory, disk, network, and port monitoring with AI-parsed natural language commands.
Supports multiple commands in one request, including per-application memory monitoring.
"""

import psutil
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import ChatOpenAI
from config.settings import Settings
from src.utils.logger import get_logger
import time

logger = get_logger(__name__)


class SystemMonitorAction(BaseModel):
    """AI-parsed system monitoring action"""
    action: str = Field(
        description="Operation type: cpu_usage, memory_usage, disk_usage, network_usage, port_status"
    )
    target: Optional[str] = Field(default=None, description="Optional target, e.g., application name or port")


class SystemMonitorActionList(BaseModel):
    """Wrapper for multiple actions"""
    actions: List[SystemMonitorAction]


# LLM 输出解析器
system_monitor_parser = PydanticOutputParser(pydantic_object=SystemMonitorActionList)

# Prompt 模板
system_monitor_prompt = PromptTemplate(
    template="""
You are a system monitoring assistant. Extract structured actions from the user's instruction.

Allowed actions:
- cpu_usage
- memory_usage
- disk_usage
- network_usage
- port_status

Rules:
- If the user mentions a specific application or process for memory, use 'target' to specify it.
- For ports, use 'target' to specify port number.
- Return a JSON object with a list of actions.

User command: {command}

{format_instructions}
""",
    input_variables=["command"],
    partial_variables={"format_instructions": system_monitor_parser.get_format_instructions()},
)


class SystemMonitorTool(BaseTool):
    """System monitoring LangChain tool (AI-driven, multi-command)"""

    name: str = "SystemMonitorTool"
    description: str = (
        "Tool for checking system CPU, memory, disk, network, and port usage. "
        "Supports multiple commands in one request, including per-application memory."
    )
    args_schema: Optional[BaseModel] = None

    def __init__(self):
        super().__init__()

    def _get_llm(self) -> ChatOpenAI:
        """Create ChatOpenAI instance"""
        settings = Settings()
        openai_config = settings.get_openai_config()
        return ChatOpenAI(
            model=openai_config.get('model', 'gpt-3.5-turbo'),
            temperature=openai_config.get('temperature', 0),
            openai_api_key=openai_config.get('api_key'),
            openai_api_base=openai_config.get('base_url'),
            max_tokens=openai_config.get('max_tokens', 2000),
        )

    def _parse_command(self, command: str) -> List[SystemMonitorAction]:
        """
        Parse natural language command(s) into structured actions using LLM.
        Example:
            "Check memory of nginx and CPU usage"
            -> [{"action": "memory_usage", "target": "nginx"}, {"action": "cpu_usage"}]
        """
        try:
            llm = self._get_llm()
            chain = system_monitor_prompt | llm | system_monitor_parser
            parsed_list: SystemMonitorActionList = chain.invoke({"command": command})
            return parsed_list.actions
        except Exception as e:
            logger.warning(f"AI command parsing failed: {e}")
            # fallback: treat whole command as unknown
            return [SystemMonitorAction(action="unknown")]

    def _run(self, command: str) -> str:
        """Execute all parsed actions and aggregate results"""
        parsed_actions = self._parse_command(command)
        results = []

        for act in parsed_actions:
            action, target = act.action, act.target
            try:
                if action == "cpu_usage":
                    results.append(self._get_cpu_usage())
                elif action == "memory_usage":
                    if target:
                        results.append(self._get_app_memory_usage(target))
                    else:
                        results.append(self._get_memory_usage())
                elif action == "disk_usage":
                    results.append(self._get_disk_usage())
                elif action == "network_usage":
                    results.append(self._get_network_usage())
                elif action == "port_status":
                    results.append(self._get_port_status(target))
                else:
                    results.append(f"❌ Unsupported system monitoring action: {action}")
            except Exception as e:
                logger.error(f"Action '{action}' failed: {e}")
                results.append(f"❌ Action '{action}' failed: {str(e)}")

        return "\n\n".join(results)

    # ---- CPU 使用情况 ----
    def _get_cpu_usage(self, top_n: int = 5) -> str:
        """
        Get system CPU usage and per-process CPU usage ranking.
        Returns top N CPU-consuming processes.
        """
        # 先初始化每个进程 CPU 统计
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                proc.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        time.sleep(1)  # 等待 1 秒统计 CPU 使用

        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            try:
                cpu = proc.cpu_percent(interval=None)
                processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cpu': cpu
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        top_cpu = sorted(processes, key=lambda x: x['cpu'], reverse=True)[:top_n]
        cpu_summary = f"System CPU Usage: {psutil.cpu_percent()}%\nPer-core: {psutil.cpu_percent(percpu=True)}%\n\nTop {top_n} CPU-consuming processes:"
        for p in top_cpu:
            cpu_summary += f"\n- {p['name']} (PID {p['pid']}): {p['cpu']}% CPU"

        return cpu_summary

    # ---- 内存使用情况 ----
    def _get_memory_usage(self, top_n: int = 5) -> str:
        """
        Get system memory usage and per-process memory usage ranking.
        Returns top N memory-consuming processes.
        """
        mem = psutil.virtual_memory()
        mem_summary = (
            f"System Memory Usage: {mem.percent}%\n"
            f"Total: {mem.total / (1024 ** 3):.2f}GB, Used: {mem.used / (1024 ** 3):.2f}GB, Free: {mem.available / (1024 ** 3):.2f}GB\n\n"
            f"Top {top_n} memory-consuming processes:"
        )

        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                mem_used = proc.info['memory_info'].rss / (1024 ** 2)  # MB
                processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'mem': mem_used
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        top_mem = sorted(processes, key=lambda x: x['mem'], reverse=True)[:top_n]
        for p in top_mem:
            mem_summary += f"\n- {p['name']} (PID {p['pid']}): {p['mem']:.2f} MB"

        return mem_summary

    # ---- Application-specific memory monitoring ----
    def _get_app_memory_usage(self, app_name: str) -> str:
        """
        Get memory usage for a specific application/process by name.
        """
        results = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                if app_name.lower() in proc.info['name'].lower():
                    mem = proc.info['memory_info'].rss  # bytes
                    results.append(
                        f"Process: {proc.info['name']} (PID: {proc.info['pid']}), Memory: {mem / (1024**2):.2f} MB"
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if not results:
            return f"No process found matching '{app_name}'"

        return "\n".join(results)

    def _get_disk_usage(self) -> str:
        parts = psutil.disk_partitions()
        results = []
        for p in parts:
            usage = psutil.disk_usage(p.mountpoint)
            results.append(
                f"{p.device} ({p.mountpoint}): {usage.percent}% used, Total: {usage.total / (1024**3):.2f}GB"
            )
        return "\n".join(results)

    def _get_network_usage(self) -> str:
        net_io = psutil.net_io_counters(pernic=True)
        results = []
        for iface, stats in net_io.items():
            results.append(
                f"{iface}: Sent: {stats.bytes_sent / (1024**2):.2f}MB, Recv: {stats.bytes_recv / (1024**2):.2f}MB"
            )
        return "\n".join(results)

    def _get_port_status(self, port: Optional[str] = None) -> str:
        results = []
        for conn in psutil.net_connections(kind='inet'):
            laddr = f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else ""
            raddr = f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "-"
            status = conn.status
            pid = conn.pid or "-"
            results.append(f"Local: {laddr}, Remote: {raddr}, Status: {status}, PID: {pid}")

        if port:
            results = [r for r in results if f":{port}" in r]

        if not results:
            return f"No process is using port {port}" if port else "No active network connections found."

        return "\n".join(results[:50]) + ("\n... (truncated)" if len(results) > 50 else "")
