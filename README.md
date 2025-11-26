# AI Agent for DevOps

一个基于Python和LangChain的智能DevOps助手，通过AI对话界面自动化和简化运维任务。

## 🚀 功能特性

### 核心功能模块

- **🐳 Docker操作**: 容器管理、镜像操作、日志查看
- **📊 系统监控**: CPU、内存、磁盘、网络状态监控
- **📁 文件管理**: 文件操作、目录管理、内容搜索
- **📋 日志分析**: 智能日志解析、错误检测、报告生成
- **⚡ 性能分析**: Python程序性能瓶颈分析
- **🔧 服务管理**: 服务状态检查、自动重启

### AI能力

- 自然语言交互界面
- 智能任务理解和执行
- 上下文感知的对话
- 自动化运维建议

## 📋 系统要求

- Python >3.8 <3.12 (推荐 3.10 以获得最佳兼容性)
- Docker (可选，用于容器操作)
- Linux/macOS/Windows (WSL2兼容)

**虚拟环境推荐**: 使用Python 3.10创建虚拟环境，确保依赖兼容。
```bash
# Ubuntu/WSL: 安装Python 3.10
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# 创建虚拟环境
python3.10 -m venv venv
source venv/bin/activate
```

## 🛠️ 安装指南

### WSL2 (Ubuntu) 特殊配置

如果在WSL2中使用，确保：

1. **Docker Desktop WSL集成**:
   - 安装Docker Desktop (Windows)
   - 在Docker Desktop设置中启用WSL2集成
   - 重启WSL: `wsl --shutdown` (在PowerShell中)
   - 在WSL中测试: `docker --version`

2. **环境变量** (可选，如果默认socket不工作):
   ```bash
   export DOCKER_HOST=tcp://localhost:2375
   ```

3. **系统依赖**:
   ```bash
   sudo apt update
   sudo apt install docker.io psmisc
   ```

### 1. 克隆项目

```bash
git clone <repository-url>
cd AIOpsAgent
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS/WSL
# 或
venv\Scripts\activate     # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境

复制配置文件并设置API密钥：

```bash
cp config/config.yaml.example config/config.yaml
```

编辑 `config/config.yaml` 文件，添加你的OpenAI API密钥：

```yaml
openai:
  api_key: "your-openai-api-key-here"
  model: "gpt-3.5-turbo"
```

或者创建 `.env` 文件：

```bash
OPENAI_API_KEY=your-openai-api-key-here
```

## 🚀 快速开始

### 启动AI Agent

```bash
python main.py
```

### 基本使用示例

```bash
# 启动交互式对话
$ python main.py

AI Agent > 你好！我是DevOps AI助手，我可以帮你：
- 管理Docker容器
- 监控系统资源
- 分析日志文件
- 检查服务状态
- 管理文件系统

请告诉我你需要什么帮助？

用户 > 帮我查看当前系统的CPU和内存使用情况

AI Agent > 正在检查系统资源状态...
CPU使用率: 45.2%
内存使用率: 68.7% (已用 5.5GB / 总共 8GB)
磁盘使用率: 72.1%

用户 > 启动名为web-app的Docker容器

AI Agent > 正在启动Docker容器 'web-app'...
✅ 容器启动成功
容器ID: abc123def456
状态: Running
端口映射: 80:8080
```

## 📁 项目结构

```
AIOpsAgent/
├── main.py                 # 主程序入口
├── requirements.txt        # Python依赖
├── .gitignore             # Git忽略文件
├── README.md              # 项目文档
├── .env.example           # 环境变量示例
├── config/                # 配置文件目录
│   ├── __init__.py
│   ├── config.yaml        # 主配置文件
│   └── settings.py        # 配置管理
├── src/                   # 源代码目录
│   ├── __init__.py
│   ├── agent/             # AI Agent核心
│   │   ├── __init__.py
│   │   └── core.py        # Agent主逻辑
│   ├── modules/           # 功能模块
│   │   ├── __init__.py
│   │   ├── docker_ops.py  # Docker操作
│   │   ├── system_monitor.py # 系统监控
│   │   ├── file_manager.py   # 文件管理
│   │   ├── log_analyzer.py   # 日志分析
│   │   └── service_check.py  # 服务检查
│   └── utils/             # 工具函数
│       ├── __init__.py
│       ├── logger.py      # 日志工具
│       ├── helpers.py     # 辅助函数
│       └── exceptions.py  # 自定义异常
├── tests/                 # 测试文件
│   ├── __init__.py
│   ├── test_agent.py
│   └── test_modules.py
└── logs/                  # 日志文件目录
```

## 🔧 配置说明

### config/config.yaml

```yaml
# AI配置
openai:
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 2000

# 系统监控配置
monitoring:
  cpu_threshold: 80.0
  memory_threshold: 85.0
  disk_threshold: 90.0
  check_interval: 30

# Docker配置
docker:
  socket_path: "unix://var/run/docker.sock"
  timeout: 30

# 日志配置
logging:
  level: "INFO"
  file: "logs/agent.log"
  max_size: "10MB"
  backup_count: 5
```

## 📖 使用指南

### Docker操作

```python
# 查看容器状态
"显示所有Docker容器的状态"

# 启动容器
"启动名为nginx的容器"

# 查看容器日志
"显示web-app容器的最新日志"
```

### 系统监控

```python
# 系统资源监控
"检查系统资源使用情况"

# 进程监控
"显示占用CPU最高的进程"

# 磁盘空间
"检查磁盘空间使用情况"
```

### 文件管理

```python
# 文件搜索
"在/var/log目录下搜索包含error的文件"

# 文件操作
"创建一个名为backup的目录"

# 内容查看
"显示nginx.conf文件的内容"
```

## 🧪 测试

运行测试套件：

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_agent.py

# 运行测试并显示覆盖率
python -m pytest tests/ --cov=src
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📝 更新日志

### v1.0.0 (2024-01-XX)
- 初始版本发布
- 实现基础AI Agent功能
- 支持Docker操作
- 系统监控功能
- 文件管理功能

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🆘 支持

如果你遇到问题或有建议，请：

1. 查看 [Issues](../../issues) 页面
2. 创建新的 Issue
3. 联系维护者

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - AI应用开发框架
- [OpenAI](https://openai.com/) - GPT模型支持
- [Docker](https://www.docker.com/) - 容器化技术
- [psutil](https://github.com/giampaolo/psutil) - 系统监控库
