"""
System monitoring module
Provides CPU, memory, disk, network, and port monitoring with AI-parsed commands.
"""

import psutil
from typing import Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import ChatOpenAI
from config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SystemMonitorAction(BaseModel):
    """AI-parsed system monitoring action"""
    action: str = Field(
        description="Operation type: cpu_usage, memory_usage, disk_usage, network_usage, port_status"
    )
    target: Optional[str] = Field(default=None, description="Optional target, e.g., port number")


system_monitor_parser = PydanticOutputParser(pydantic_object=SystemMonitorAction)

system_monitor_prompt = PromptTemplate(
    template=
    """
    You are a system monitoring parsing assistant. Extract structured fields from the user's instruction.

    Allowed actions (must exactly match):
    - cpu_usage
    - memory_usage
    - disk_usage
    - network_usage
    - port_status

    Rules:
    - target: Optional, e.g., port number. Use None if not specified.

    Command: {command}

    {format_instructions}
    """,
    input_variables=["command"],
    partial_variables={"format_instructions": system_monitor_parser.get_format_instructions()},
)


class SystemMonitorTool(BaseTool):
    """System monitoring LangChain tool"""

    name: str = "system_monitor"
    description: str = (
        "Tool for checking system CPU, memory, disk, network, and port usage. "
        "Input should be natural language, like 'show CPU usage' or 'check port 8080 usage'."
    )
    args_schema: Optional[BaseModel] = None

    def __init__(self):
        super().__init__()

    def _parse_command(self, command: str) -> SystemMonitorAction:
        """Use LLM to parse command into action + target"""
        settings = Settings()
        openai_config = settings.get_openai_config()
        llm = ChatOpenAI(
            model=openai_config.get('model', 'gpt-3.5-turbo'),
            temperature=openai_config.get('temperature', 0),
            openai_api_key=openai_config.get('api_key'),
            openai_api_base=openai_config.get('base_url'),
            max_tokens=openai_config.get('max_tokens', 2000),
        )
        chain = system_monitor_prompt | llm | system_monitor_parser
        try:
            parsed: SystemMonitorAction = chain.invoke({"command": command})
            return parsed
        except Exception as e:
            logger.warning(f"AI command parsing failed: {e}")
            return SystemMonitorAction(action="unknown")

    def _run(self, command: str) -> str:
        """Execute the system monitoring action"""
        parsed = self._parse_command(command)
        action, target = parsed.action, parsed.target

        if action == "cpu_usage":
            return self._get_cpu_usage()
        elif action == "memory_usage":
            return self._get_memory_usage()
        elif action == "disk_usage":
            return self._get_disk_usage()
        elif action == "network_usage":
            return self._get_network_usage()
        elif action == "port_status":
            return self._get_port_status(target)
        else:
            return f"Unsupported system monitoring action: {action}"

    def _get_cpu_usage(self) -> str:
        cpu_percent = psutil.cpu_percent(interval=1)
        per_core = psutil.cpu_percent(interval=1, percpu=True)
        return f"CPU Usage: {cpu_percent}%\nPer core: {per_core}"

    def _get_memory_usage(self) -> str:
        mem = psutil.virtual_memory()
        return (
            f"Memory Usage: {mem.percent}%\n"
            f"Total: {mem.total / (1024**3):.2f}GB, Used: {mem.used / (1024**3):.2f}GB, Free: {mem.available / (1024**3):.2f}GB"
        )

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


if __name__ == "__main__":
    tool = SystemMonitorTool()
    print("CPU Usage:")
    print(tool._run("show cpu usage"))
    print("\nPort 22 usage:")
    print(tool._run("check port 22 usage"))
