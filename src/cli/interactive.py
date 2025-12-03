# src/cli/interactive.py

"""
Interactive chat loop for AI DevOps Agent.
"""
from rich.console import Console
from rich.prompt import Prompt
from src.utils.logger import get_logger

console = Console()
logger = get_logger("AIOpsAgent.interactive")


def interactive_chat(agent):
    """Run an interactive chat session with the agent."""
    console.print("[bold yellow]Enter interactive mode...[/bold yellow]")

    while True:
        user_input = Prompt.ask("[bold]You[/bold]")

        if user_input.lower() in ["exit", "quit"]:
            console.print("[bold green]Goodbye![/bold green]")
            break

        with console.status("[bold green]AI thinking...[/bold green]"):
            result = agent.chat(user_input)

        console.print(f"[bold]You[/bold]: {user_input}")
        console.print(f"[bold cyan]AI[/bold cyan]: {result['response']}")
