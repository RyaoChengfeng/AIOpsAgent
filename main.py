"""
AI Agent for DevOps - 主程序入口
基于命令行的智能DevOps助手
"""

import sys
import signal
import os
from datetime import datetime
from typing import Dict, Any
import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import print as rprint

from config.settings import get_settings, validate_config
from src.agent.core import create_devops_agent
from src.utils.logger import setup_logger, get_logger
from src.utils.helpers import format_bytes, truncate_string, get_system_info
from src.utils.exceptions import ConfigurationError, AIAgentError

# 设置Rich控制台
console = Console()

# 设置日志
logger = setup_logger("AIOpsAgent.main")


@click.group(invoke_without_command=True)
@click.version_option("1.0.0", "-v", "--version")
@click.option('--debug', is_flag=True, help='启用调试模式')
def cli(debug: bool):
    """
    AI Agent for DevOps - 智能运维助手

    通过自然语言与AI对话，自动化管理DevOps任务。
    支持Docker操作、系统监控、文件管理、日志分析、服务检查等功能。
    """
    if debug:
        logger.setLevel("DEBUG")
        console.print("[bold green]调试模式已启用[/bold green]")
    
    if not validate_config():
        console.print("[bold red]配置验证失败！请检查 config/config.yaml 和环境变量。[/bold red]")
        sys.exit(1)
    
    # WSL2检测
    if os.uname().sysname.lower() == 'linux' and 'microsoft' in os.uname().release.lower():
        console.print("[bold yellow]检测到WSL2环境。确保Docker Desktop WSL集成已启用。[/bold yellow]")
    
    if click.Context(cli).invoked_subcommand is None:
        # 启动交互式模式
        agent = create_devops_agent()
        interactive_chat(agent)


@cli.command()
@click.option('--file', '-f', type=click.Path(exists=True), help='从文件读取输入')
def chat(file: str):
    """与AI Agent进行对话"""
    agent = None
    try:
        agent = create_devops_agent()
        console.print(Panel.fit(
            "[bold cyan]欢迎使用 AI Agent for DevOps！[/bold cyan]\n"
            "我可以帮助你:\n"
            "• 管理Docker容器和镜像\n"
            "• 监控系统资源 (CPU/内存/磁盘)\n"
            "• 操作文件和目录\n"
            "• 分析日志文件\n"
            "• 检查和管理系统服务\n\n"
            "[bold green]输入 'exit' 或 'quit' 退出，'help' 查看帮助。[/bold green]",
            title="🤖 AI DevOps 助手",
            border_style="blue"
        ))
        
        if file:
            with open(file, 'r', encoding='utf-8') as f:
                user_input = f.read().strip()
                if user_input:
                    result = agent.chat(user_input)
                    console.print(f"[bold]用户: [/bold]{user_input}")
                    console.print(f"[bold cyan]AI: [/bold cyan]{result['response']}")
                    if not result['success']:
                        console.print(f"[bold red]错误: {result.get('error', '未知错误')}[/bold red]")
        else:
            interactive_chat(agent)
            
    except ConfigurationError as e:
        console.print(f"[bold red]配置错误: {e}[/bold red]")
        sys.exit(1)
    except AIAgentError as e:
        console.print(f"[bold red]AI Agent错误: {e}[/bold red]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]用户中断。[/bold yellow]")
    except Exception as e:
        logger.error(f"程序执行错误: {e}")
        console.print(f"[bold red]意外错误: {e}[/bold red]")
    finally:
        if agent:
            agent.shutdown()


@cli.command()
def status():
    """显示系统和Agent状态"""
    try:
        settings = get_settings()
        system_info = get_system_info()
        
        table = Table(title="系统和Agent状态")
        table.add_column("项目", style="cyan")
        table.add_column("状态/信息", style="magenta")
        
        table.add_row("配置加载", "✅ 成功" if validate_config() else "❌ 失败")
        table.add_row("OpenAI API", "已配置" if settings.get('openai.api_key') else "未配置")
        table.add_row("操作系统", f"{system_info.get('system', '未知')} {system_info.get('release', '')}")
        table.add_row("主机名", system_info.get('hostname', '未知'))
        table.add_row("CPU核心", str(system_info.get('cpu_count', '未知')))
        table.add_row("总内存", format_bytes(system_info.get('memory_total', 0)))
        table.add_row("Python版本", system_info.get('python_version', '未知'))
        table.add_row("Agent版本", settings.get('app.version', '1.0.0'))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]获取状态失败: {e}[/bold red]")


@cli.command()
def tools():
    """列出可用工具"""
    try:
        agent = create_devops_agent()
        tools_info = agent.get_available_tools()
        
        table = Table(title="可用工具列表")
        table.add_column("工具名称", style="cyan")
        table.add_column("描述", style="magenta")
        
        for tool in tools_info:
            table.add_row(tool['name'], truncate_string(tool['description'], 60))
        
        console.print(table)
        agent.shutdown()
        
    except Exception as e:
        console.print(f"[bold red]获取工具列表失败: {e}[/bold red]")


def interactive_chat(agent):
    """交互式聊天循环"""
    console.print("[bold yellow]开始交互模式...[/bold yellow]")
    
    while True:
        try:
            user_input = Prompt.ask("[bold]你[/bold]", console=console)
            
            if user_input.lower() in ['exit', 'quit', '退出', '结束']:
                console.print("[bold green]再见！[/bold green]")
                break
            elif user_input.lower() in ['help', '帮助']:
                console.print("""
[bold]可用命令:[/bold]
• Docker: "启动web容器" "查看容器日志"
• 系统监控: "检查CPU使用" "显示Top进程"
• 文件管理: "创建config.txt" "列出当前目录"
• 日志分析: "分析error.log错误" "搜索数据库错误"
• 服务检查: "检查nginx状态" "重启mysql服务"
• 通用: "系统状态" "可用工具"

输入 'exit' 退出。
                """)
                continue
            elif not user_input.strip():
                continue
            
            with console.status("[bold green]AI正在思考...[/bold green]"):
                result = agent.chat(user_input)
            
            console.print(f"[bold]你[/bold]: {user_input}")
            
            if result['success']:
                console.print(f"[bold cyan]AI[/bold cyan]: {result['response']}")
                
                # 如果有中间步骤，显示工具使用
                if result.get('intermediate_steps'):
                    console.print("[dim italic]工具执行记录:[/dim italic]")
                    for step in result['intermediate_steps'][:3]:  # 显示前3个步骤
                        if isinstance(step, list) and len(step) >= 2:
                            tool_name = step[0] if isinstance(step[0], str) else str(step[0])
                            tool_output = str(step[1])[:200] + "..." if len(str(step[1])) > 200 else str(step[1])
                            console.print(f"  📦 {tool_name}: {tool_output}")
            else:
                console.print(f"[bold red]AI[/bold red]: {result['response']}")
                if result.get('error'):
                    console.print(f"[bold red]错误详情: {result['error']}[/bold red]")
            
            console.print()  # 空行分隔
            
        except KeyboardInterrupt:
            console.print("\n[bold yellow]对话已中断。[/bold yellow]")
            break
        except EOFError:
            console.print("\n[bold yellow]输入结束。[/bold yellow]")
            break
        except Exception as e:
            logger.error(f"交互错误: {e}")
            console.print(f"[bold red]对话出错: {e}[/bold red]")
            continue


def signal_handler(sig, frame):
    """信号处理器"""
    console.print("\n[bold yellow]程序正在退出...[/bold yellow]")
    sys.exit(0)


if __name__ == "__main__":
    # 设置信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    
    # 显示欢迎信息
    console.print(Panel(
        "[bold cyan]AI Agent for DevOps v1.0.0[/bold cyan]\n"
        "智能运维助手 - 基于LangChain和OpenAI\n\n"
        "[dim]使用 'python main.py --help' 查看命令[/dim]",
        title="🚀 启动成功",
        border_style="green"
    ))
    
    # 启动CLI
    cli()
