# src/cli/commands/status_cmd.py
import click
from rich.console import Console
from rich.table import Table

from src.cli import cli
from config.settings import get_settings, validate_config
from src.utils.helpers import format_bytes, truncate_string, get_system_info
from src.utils.logger import get_logger

console = Console()
logger = get_logger("AIOpsAgent.status")

@cli.command(name="status")
def status_cmd():
    """Display system and AI Agent status"""
    try:
        settings = get_settings()           # Load configuration once
        system_info = get_system_info()     # Gather system info once

        table = Table(title="System and AI Agent Status")
        table.add_column("Item", style="cyan")
        table.add_column("Status/Info", style="magenta")

        table.add_row("Config Load", "Success" if validate_config() else "Failed")
        table.add_row("OpenAI API", "Configured" if settings.get('openai.api_key') else "Not Configured")
        table.add_row("OS", system_info.get('system', 'Unknown'))
        table.add_row("CPU Cores", str(system_info.get('cpu_count', 'Unknown')))
        table.add_row("Total Memory", format_bytes(system_info.get('memory_total', 0)))

        console.print(table)
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        console.print(f"[bold red]Error retrieving system status:[/bold red] {e}")

    