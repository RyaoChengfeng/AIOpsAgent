# src/cli/commands/tools_cmd.py
import click
from rich.table import Table
from rich.console import Console
from src.cli import cli
from src.agent.core import create_devops_agent
from src.utils.helpers import truncate_string
from src.utils.logger import get_logger

console = Console()
logger = get_logger("AIOpsAgent.tools")

@cli.command(name="tools")
def tools_cmd():
    """List available tools"""
    agent = None
    try:
        agent = create_devops_agent()
        tools_info = agent.get_available_tools()

        table = Table(title="Available Tools")
        table.add_column("Tool Name", style="cyan")
        table.add_column("Description", style="magenta")

        for tool in tools_info:
            table.add_row(tool["name"], truncate_string(tool["description"], 40))

        console.print(table)

    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        console.print(f"[bold red]Error listing toos:[/bold red] {e}")

    finally:
        if agent:
            agent.shutdown()