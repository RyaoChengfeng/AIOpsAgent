"""
Service Checking Module (AI-driven)
Provides utility functions and a LangChain tool for checking and managing system services.
Supports multiple actions in one request and non-blocking service operations.
"""

import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

import psutil
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

from config.settings import get_config
from config.settings import Settings
from src.utils.logger import get_logger
from src.utils.helpers import (
    run_command,
    is_port_open,
    get_process_by_name,
    get_process_by_port,
    format_bytes,
)
from src.utils.exceptions import ServiceCheckError, CommandExecutionError

logger = get_logger(__name__)


class ServiceConfig(BaseModel):
    """Service checking configuration model"""
    timeout: int = Field(default_factory=lambda: get_config('service_check.timeout', 10))
    retry_count: int = Field(default_factory=lambda: get_config('service_check.retry_count', 3))
    retry_delay: int = Field(default_factory=lambda: get_config('service_check.retry_delay', 5))


class ServiceAction(BaseModel):
    """
    Pydantic model for parsed service commands.
    - action: one of allowed action strings
    - service_name: optional service / process name
    - port: optional port number
    - args: optional dict for additional parameters
    """
    action: str = Field(description="Action type. Must be one of the allowed actions.")
    service_name: Optional[str] = Field(default=None, description="Service or process name (e.g., nginx, mysql)")
    port: Optional[int] = Field(default=None, description="Port number (e.g., 80, 3306)")
    args: Optional[Dict[str, Any]] = Field(default=None, description="Additional action-specific arguments")


class ServiceActionList(BaseModel):
    """Wrapper to support multiple actions in one request"""
    actions: List[ServiceAction]


# Setup parser and prompt
service_parser = PydanticOutputParser(pydantic_object=ServiceActionList)

ALLOWED_ACTIONS = [
    "check_service_status",
    "restart_service",
    "start_service",
    "stop_service",
    "check_port",
    "list_services"
]

service_prompt = PromptTemplate(
    template="""
You are a professional service-management command parser. Convert the user's natural language request
into a strict JSON structure that matches the provided Pydantic model.

Allowed actions (choose from):
{actions_list}

Field rules:
- action: one of the allowed actions.
- service_name: the service or process name if applicable (string). Use null if not provided.
- port: integer port number if applicable (1-65535). Use null if not provided.
- args: an optional object for other parameters (use null if none).

If the user didn't mention required parameters, set them to null rather than guessing.
Output MUST follow the exact format instructions below.

User request: {request}

{format_instructions}
""",
    input_variables=["request"],
    partial_variables={
        "actions_list": "\n- ".join([""] + ALLOWED_ACTIONS),
        "format_instructions": service_parser.get_format_instructions()
    },
)


class ServiceCheckerTool(BaseTool):
    """Service checking LangChain tool (AI-driven parsing, multi-command support)."""

    name: str = "service_checker"
    description: str = (
        "A tool to check and manage system services. "
        "Supports multiple actions in one request. "
        "Commands include: 'check nginx service status', 'restart mysql', 'start apache2', "
        "'stop redis', 'check port 3306', 'list running services', etc."
    )
    args_schema: Optional[BaseModel] = None

    def __init__(self):
        super().__init__()

    def _get_llm(self) -> ChatOpenAI:
        """Create ChatOpenAI instance from Settings."""
        settings = Settings()
        openai_config = settings.get_openai_config()
        max_tokens = openai_config.get('max_tokens', 2000)
        model = openai_config.get('model', 'gpt-3.5-turbo')
        temperature = openai_config.get('temperature', 0)
        api_key = openai_config.get('api_key')
        base_url = openai_config.get('base_url')
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=base_url,
            max_tokens=max_tokens,
            default_headers={
                "HTTP-Referer": "https://localhost/",
                "X-Title": "ServiceChecker-AI-Tool"
            }
        )

    def _parse_command(self, request: str) -> List[ServiceAction]:
        """
        Parse a natural language request into a list of ServiceAction using LLM + Pydantic parser.
        Returns a list of ServiceAction. On parse failure, returns a fallback with action 'unknown'.
        """
        llm = self._get_llm()
        chain = service_prompt | llm | service_parser
        try:
            parsed_list: ServiceActionList = chain.invoke({"request": request})
            return parsed_list.actions
        except Exception as e:
            logger.warning(f"AI command parsing failed: {e}")
            return [ServiceAction(action="unknown", service_name=None, port=None, args=None)]

    def _run(self, request: str) -> str:
        """
        Entry point: parse the request and dispatch to each action handler.
        Supports multiple actions.
        """
        try:
            parsed_actions: List[ServiceAction] = self._parse_command(request)

            results = []
            for parsed in parsed_actions:
                action = parsed.action
                try:
                    if action == "check_service_status":
                        results.append(self._check_service_status(parsed.service_name))
                    elif action == "restart_service":
                        results.append(self._restart_service(parsed.service_name))
                    elif action == "start_service":
                        results.append(self._start_service(parsed.service_name))
                    elif action == "stop_service":
                        results.append(self._stop_service(parsed.service_name))
                    elif action == "check_port":
                        results.append(self._check_port_service(parsed.port))
                    elif action == "list_services":
                        results.append(self._list_all_services())
                    else:
                        results.append(f"❌ Unsupported or unknown action: '{action}'")
                except Exception as e:
                    logger.error(f"Action '{action}' failed: {e}")
                    results.append(f"❌ Action '{action}' failed: {str(e)}")

            return "\n\n".join(results)

        except Exception as e:
            logger.error(f"Service check execution failed: {e}")
            raise ServiceCheckError(f"Service checking execution failed: {str(e)}")

    # ---- Action handlers (reuse your existing logic) ----

    def _check_service_status(self, service_name: Optional[str]) -> str:
        """Check service status via systemctl or by scanning processes if not a systemd unit."""
        config = ServiceConfig()

        if not service_name:
            return "Please specify a service name to check (e.g., 'nginx' or 'mysql')."

        try:
            result = run_command(f"systemctl is-active {service_name}", timeout=config.timeout)
            if result[0] == 0:
                status = result[1].strip()
                return self._get_detailed_service_info(service_name, status)
            elif "not-found" in (result[2] or "").lower():
                # fall back to scanning processes
                return self._check_by_process(service_name)
            else:
                return f"❌ Failed to check service '{service_name}': {result[2] or 'Unknown error'}"
        except Exception as e:
            logger.error(f"Failed to check service status: {e}")
            raise ServiceCheckError(f"Unable to check service '{service_name}' status: {str(e)}")

    def _get_detailed_service_info(self, service_name: str, status: str) -> str:
        """Get detailed unit status using systemctl status."""
        try:
            result = run_command(f"systemctl status {service_name} --no-pager", timeout=10)
            if result[0] == 0:
                output = result[1] or ""
                lines = output.splitlines()
                info_lines = []
                # pick top relevant lines
                for line in lines[:20]:
                    low = line.lower()
                    if any(k in low for k in ['active', 'loaded', 'main pid', 'since', 'cpu', 'memory']):
                        info_lines.append(line.rstrip())
                detailed_info = "\n".join(info_lines).strip()
                result_str = f"Service '{service_name}' status: {status}\n"
                result_str += "=" * 40 + "\n"
                result_str += (detailed_info or "(no detailed info extracted)")
                if status != "active":
                    result_str += f"\n\nTip: Service '{service_name}' is not active. You may try 'restart {service_name}'."
                return result_str
            else:
                return f"Unable to get detailed info for '{service_name}': {result[2] or 'Unknown error'}"
        except Exception as e:
            logger.error(f"Failed to get detailed service info: {e}")
            return f"Service '{service_name}' status: {status} (detailed info unavailable: {str(e)})"

    def _check_by_process(self, name: str) -> str:
        """Check processes that match a name (fallback when systemd unit not found)."""
        try:
            processes = get_process_by_name(name)
            if not processes:
                return f"❌ No processes matching '{name}' found. The service may not be running."

            result = f"Processes related to '{name}' (top results):\n"
            result += "-" * 40 + "\n"
            for proc in processes[:10]:
                pid = proc.get('pid')
                pname = proc.get('name', '<unknown>')
                cpu = proc.get('cpu_percent', 0.0)
                mem = proc.get('memory_percent', 0.0)
                result += f"PID: {pid}, Name: {pname}\n"
                result += f"  CPU: {cpu:.1f}%, Memory: {mem:.1f}%\n"
                # try to get additional info from psutil
                try:
                    ps = psutil.Process(pid)
                    cmdline = " ".join(ps.cmdline()) if ps.cmdline() else "(no command line)"
                    start = datetime.fromtimestamp(ps.create_time()).strftime('%Y-%m-%d %H:%M:%S')
                    rss = format_bytes(ps.memory_info().rss)
                    result += f"  Cmd: {cmdline}\n"
                    result += f"  Start: {start}, RSS: {rss}\n\n"
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    result += "  (Insufficient permissions to obtain more details)\n\n"
            return result
        except Exception as e:
            logger.error(f"Process fallback check failed: {e}")
            return f"Failed to check processes for '{name}': {str(e)}"

    def _list_all_services(self) -> str:
        """List running systemd services or fall back to process count if systemctl not available."""
        try:
            result = run_command("sudo systemctl list-units --type=service --state=running --no-pager", timeout=15)
            if result[0] == 0:
                output = result[1] or ""
                lines = output.splitlines()
                running_services = []
                for line in lines:
                    stripped = line.strip()
                    if not stripped or stripped.startswith('UNIT') or stripped.startswith('LOAD') or stripped.startswith('ACTIVE'):
                        continue
                    parts = stripped.split()
                    if parts:
                        unit_name = parts[0]
                        if unit_name.endswith('.service'):
                            running_services.append(unit_name.replace('.service', ''))
                result_str = f"Running services ({len(running_services)}):\n"
                result_str += "-" * 40 + "\n"
                for svc in running_services[:50]:
                    result_str += f"- {svc}\n"
                if len(running_services) > 50:
                    result_str += f"... and {len(running_services)-50} more\n"
                return result_str
            else:
                # fallback: list top processes count
                ps_result = run_command("ps aux --no-headers | wc -l", timeout=5)
                if ps_result[0] == 0:
                    process_count = int(ps_result[1].strip())
                    return f"Total running processes: {process_count}\n(systemd list unavailable)"
                else:
                    return "Unable to list services or processes on this host."
        except Exception as e:
            logger.error(f"Failed to list services: {e}")
            raise ServiceCheckError(f"Unable to list services: {str(e)}")

    def _restart_service(self, service_name: Optional[str]) -> str:
        """Restart a systemd service (or notify if not specified)."""
        config = ServiceConfig()
        if not service_name:
            return "Please specify a service name to restart."

        confirmation_msg = (
            f"⚠️ Warning: About to restart service '{service_name}'. This may interrupt connections.\n"
        )

        try:
            result = run_command(f"sudo systemctl restart {service_name}", timeout=config.timeout)
            if result[0] == 0:
                # brief wait then show status
                time.sleep(2)
                status_text = self._check_service_status(service_name)
                return confirmation_msg + f"✅ Service '{service_name}' restarted successfully!\n\n{status_text}"
            else:
                return confirmation_msg + f"❌ Failed to restart '{service_name}': {result[2] or 'Unknown error'}"
        except Exception as e:
            logger.error(f"Failed to restart service: {e}")
            return confirmation_msg + f"❌ Failed to restart '{service_name}': {str(e)}"

    def _start_service(self, service_name: Optional[str]) -> str:
        """Start a systemd service."""
        config = ServiceConfig()
        if not service_name:
            return "Please specify a service name to start."

        try:
            result = run_command(f"sudo systemctl start {service_name}", timeout=config.timeout)
            if result[0] == 0:
                time.sleep(1)
                status_text = self._check_service_status(service_name)
                return f"✅ Service '{service_name}' started successfully!\n\n{status_text}"
            else:
                return f"❌ Failed to start service '{service_name}': {result[2] or 'Unknown error'}"
        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            raise ServiceCheckError(f"Unable to start service '{service_name}': {str(e)}")

    def _stop_service(self, service_name: Optional[str]) -> str:
        """Stop a systemd service."""
        config = ServiceConfig()
        if not service_name:
            return "Please specify a service name to stop."

        warning_msg = (
            f"⚠️ Warning: About to stop service '{service_name}'. This will disconnect dependent connections.\n"
        )

        try:
            result = run_command(f"sudo systemctl stop {service_name}", timeout=config.timeout)
            if result[0] == 0:
                return warning_msg + f"✅ Service '{service_name}' stopped."
            else:
                return warning_msg + f"❌ Failed to stop service '{service_name}': {result[2] or 'Unknown error'}"
        except Exception as e:
            logger.error(f"Failed to stop service: {e}")
            raise ServiceCheckError(f"Unable to stop service '{service_name}': {str(e)}")

    def _check_port_service(self, port: Optional[int]) -> str:
        """Check port status and process information for the given port."""
        if port is None:
            return "Please specify a port number to check (e.g., 80 or 3306)."

        try:
            if not (1 <= port <= 65535):
                return "Port must be in the range 1-65535."

            is_open = is_port_open('localhost', port)
            process = get_process_by_port(port)

            result = f"Port {port} service check:\n"
            result += "=" * 30 + "\n"
            result += f"Port status: {'🟢 OPEN' if is_open else '🔴 CLOSED'}\n"

            if process:
                pid = process.get('pid')
                name = process.get('name', '<unknown>')
                result += f"\nProcess using the port:\n"
                result += f"  Name: {name}\n"
                result += f"  PID: {pid}\n"
                try:
                    proc = psutil.Process(pid)
                    cmd = " ".join(proc.cmdline()) if proc.cmdline() else "(no command line)"
                    cpu = proc.cpu_percent(interval=0.1)
                    mem = format_bytes(proc.memory_info().rss)
                    start = datetime.fromtimestamp(proc.create_time()).strftime('%Y-%m-%d %H:%M:%S')
                    result += f"  Command: {cmd}\n"
                    result += f"  CPU usage: {cpu:.1f}%\n"
                    result += f"  Memory usage: {mem}\n"
                    result += f"  Start time: {start}\n"
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    result += "  (Unable to obtain detailed information - insufficient permissions)\n"

                common_ports = {
                    22: "SSH",
                    80: "HTTP (Apache/Nginx)",
                    443: "HTTPS",
                    3306: "MySQL",
                    5432: "PostgreSQL",
                    6379: "Redis",
                    27017: "MongoDB"
                }
                service_name = common_ports.get(port, "Unknown")
                result += f"\nPossible service: {service_name}"
            else:
                result += "\nNo process is currently using this port.\n"
                if is_open:
                    result += "Port is open but unused — potential security risk."
                else:
                    result += "Port is closed. No service running on this port."

            return result
        except Exception as e:
            logger.error(f"Port check failed: {e}")
            raise ServiceCheckError(f"Unable to check service on port {port}: {str(e)}")


if __name__ == "__main__":
    # Quick local test (will use parsing via LLM; ensure Settings has valid OpenAI config)
    try:
        tool = ServiceCheckerTool()
        print("Testing service checker tool (AI-driven parsing).")
        sample_requests = [
            "Check nginx service status",
            "restart mysql",
            "start apache2",
            "stop redis",
            "What is using port 3306?",
            "List all running services"
        ]
        for req in sample_requests:
            print("\n>>> Request:", req)
            print(tool._run(req))
    except Exception as e:
        print(f"Test failed: {e}")
