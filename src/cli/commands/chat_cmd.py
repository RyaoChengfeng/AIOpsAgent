# src/cli/commands/chat_cmd.py
import sys
import click
from rich.console import Console
from rich.panel import Panel

from src.cli import cli
from src.agent.core import create_devops_agent
from src.cli.interactive import interactive_chat
from src.utils.exceptions import ConfigurationError, AIAgentError
from src.utils.logger import get_logger

console = Console()
logger = get_logger("AIOpsAgent.chat")

@cli.command(name="chat")
@click.option('--file', '-f', type=click.Path(exists=True), help='Read input from a file')
def chat(file):
    """Interact with the AI Agent"""
    agent = None
    try:
        agent = create_devops_agent()
        # Welcome message
        console.print(Panel.fit(
            "[bold cyan]Welcome to AI Agent for DevOps![/bold cyan]\n"
            "- Docker management\n"
            "- System monitoring\n"
            "- File operations\n"
            "- Service management\n\n"
            "Type 'exit' to quit.",
            title="AI DevOps Assitant",
            border_style="blue"
        ))
        if file:
            with open(file, 'r', encoding='utf-8') as f:
                user_input = f.read().strip()
                if user_input:
                    result = agent.chat(user_input)
                    console.print(f"[bold]User:[/bold]{user_input}")
                    console.print(f"[bold cyan]AI:[/bold cyan]{result['response']}")
        else:
            interactive_chat(agent)

    except (ConfigurationError, AIAgentError) as e:
        console.print(f"[bold red]{e}[/bold red]")
        sys.exit(1)
    finally:
        if agent:
            agent.shutdown()