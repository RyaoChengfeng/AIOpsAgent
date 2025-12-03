# src/cli/__init__.py
"""
CLI package initialization.
Defines the main CLI group for AI DevOps Agent.
"""

import click

@click.group(invoke_without_command=True)
def cli():
    """
    Root command group for AIOpsAgent CLI
    This group acts as the main entry point for all subcommands.
    """
    pass

from .commands.chat_cmd import *
from .commands.status_cmd import *
from .commands.tools_cmd import *
