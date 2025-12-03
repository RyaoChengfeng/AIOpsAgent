# src/cli/entry.py

"""
Main CLI entry point.
Handles global checks, WSL detection, and interactive mode.
"""

import sys
import os
import signal
import platform
from rich.console import Console


from src.cli import cli
from config.settings import validate_config
from src.agent.core import create_devops_agent
from src.utils.logger import setup_logger
from src.cli.interactive import interactive_chat


# Console and logger
console = Console()
logger = setup_logger("AIOpsAgent.cli")


def _is_wsl():
    """
    Detect WSL2 environment in a cross-platform safe way.
    """
    if platform.system().lower() != 'linux':
        return False
    try:
        with open('/proc/version', 'r', encoding='utf-8') as f:
            return 'microsoft' in f.read().lower()
    except Exception:
        return False


def run_interactive_mode():
    """
    Launch interactive mode when no subcommand is provided.
    """
    if not validate_config():
        console.print("[bold red]Configuration validation failed[/bold red]")
        sys.exit(1)

    if _is_wsl():
        console.print("[bold yellow]WSL2 environment detected. Ensure Docker WSL integration is enabled.[/bold yellow]")

    # Start interactive agent session
    agent = create_devops_agent()
    interactive_chat(agent)

@cli.result_callback()
def run_interactive(*args, **kwargs):
    """Called when no subcommand is invoked"""
    run_interactive_mode()

def signal_handler(sig, frame):
    """Handle Ctrl+C signal"""
    console.print("\n[bold yellow]Exiting program...[/bold yellow]")
    sys.exit(0)


# Register Ctrl+C handler
signal.signal(signal.SIGINT, signal_handler)
