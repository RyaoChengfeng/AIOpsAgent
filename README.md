# AI Agent for DevOps

A Python-based intelligent DevOps assistant using LangChain.  
Automates and simplifies DevOps tasks via an interactive AI chat interface.

## 🚀 Features

### Core Modules

- **🐳 Docker Operations**: Container management, image operations, log viewing
- **📊 System Monitoring**: CPU, memory, disk, and network usage
- **📁 File Management**: File and directory operations, content search
- **📋 Log Analysis**: Intelligent log parsing, error detection, report generation
- **⚡ Performance Analysis**: Python program profiling
- **🔧 Service Management**: Check and restart system services

### AI Capabilities

- Natural language interactive interface
- Context-aware conversations
- Automated operational suggestions
- Intelligent task understanding and execution

## 📋 System Requirements

- Python 3.8–3.11 (recommended 3.10 for compatibility)
- Docker (optional, for container operations)
- Linux/macOS/Windows (WSL2 compatible)

**Recommended**: use a Python 3.10 virtual environment:

```bash
# Ubuntu/WSL: Install Python 3.10
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate


## 🛠️ Installation

### 1. Clone the project

```bash
git clone <repository-url>
cd AIOpsAgent
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS/WSL
# OR
venv\Scripts\activate         # Windows

```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Copy the example configuration and set your OpenAI API key:

```bash
cp config/config.yaml. config/config.local.yaml
```
and make sure put your `config.local.yaml` in `.gitignore`

Edit config/config.yaml
```bash
openai:
  api_key: "your-openai-api-key-here"
  model: "gpt-3.5-turbo"
```


## 🚀 Quick Start

### Run the AI Agent

```bash
python main.py
```

### Commands

```bash
# Start interactive chat
python main.py chat

# Check system status
python main.py status

# List available tools
python main.py tools

```


## 📁 Project Structure

```
AIOpsAgent/
├── main.py
├── requirements.txt
├── README.md
├── .env.example
├── config/
│   ├── __init__.py
│   ├── config.yaml
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   └── core.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── docker_ops.py
│   │   ├── system_monitor.py
│   │   ├── file_manager.py
│   │   ├── log_analyzer.py
│   │   └── service_check.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── entry.py
│   │   ├── interactive.py
│   │   └── commands/
│   │       ├── __init__.py
│   │       ├── chat_cmd.py
│   │       ├── status_cmd.py
│   │       └── tools_cmd.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── helpers.py
│       └── exceptions.py
└── logs/

```

## 🔧 Configuration

### config/config.yaml

```yaml
openai:
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 2000

monitoring:
  cpu_threshold: 80.0
  memory_threshold: 85.0
  disk_threshold: 90.0
  check_interval: 30

docker:
  socket_path: "unix://var/run/docker.sock"
  timeout: 30

logging:
  level: "INFO"
  file: "logs/agent.log"
  max_size: "10MB"
  backup_count: 5
```


## 📝 Changelog

v1.1.0 (2025-10-5)
- Refactored the project into a more modular and maintainable directory structure, separating agents, CLI, utilities, tools, and configuration.
- Improved the agent core architecture, isolating initialization logic, prompt templates, and execution flow for better extensibility.
- Updated and cleaned dependency definitions in requirements.txt, removing unused packages and aligning versions with the new LangChain / OpenAI SDK implementations.
- Enhanced internal error handling to provide clearer diagnostics during agent initialization and runtime.
- Introduced clearer boundaries between components such as LLM wrappers, tool definitions, and execution pipelines.

v1.0.0 (2025-9-5)
- Initial project setup.
- Implemented basic CLI.
- Added simple LLM wrapper and preliminary tool integration.



