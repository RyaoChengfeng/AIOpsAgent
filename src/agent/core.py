"""
AI Agent核心逻辑模块
实现基于LangChain的智能Agent
"""

import os
from typing import Dict, List, Any, Optional
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks import CallbackManagerForLLMRun
from config.settings import get_config
from src.utils.logger import get_logger
from src.utils.exceptions import AIAgentError, ConfigurationError
from src.modules import (
    DockerOpsTool, SystemMonitorTool, FileManagerTool,
    LogAnalyzerTool, PerformanceAnalyzerTool, ServiceCheckerTool
)

logger = get_logger(__name__)


class DevOpsAgent:
    """DevOps AI Agent主类"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化DevOps AI Agent
        
        Args:
            config_path: 配置文件路径
        """
        self.config = get_config
        self._validate_configuration()
        
        # 加载OpenAI配置
        openai_config = self.config.get_openai_config()
        api_key = openai_config.get('api_key')
        model = openai_config.get('model', 'gpt-3.5-turbo')
        temperature = openai_config.get('temperature', 0.7)
        max_tokens = openai_config.get('max_tokens', 2000)
        
        if not api_key or api_key == 'your-openai-api-key-here':
            raise ConfigurationError("请设置有效的OpenAI API密钥")
        
        os.environ['OPENAI_API_KEY'] = api_key
        
        # 初始化语言模型
        try:
            if 'gpt-3.5-turbo' in model or 'gpt-4' in model:
                self.llm = ChatOpenAI(
                    model_name=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    openai_api_key=api_key
                )
            else:
                self.llm = OpenAI(
                    model_name=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    openai_api_key=api_key
                )
            logger.info(f"成功初始化语言模型: {model}")
        except Exception as e:
            logger.error(f"初始化语言模型失败: {e}")
            raise AIAgentError(f"无法初始化AI模型: {e}")
        
        # 初始化工具
        self.tools = self._initialize_tools()
        
        # 初始化Agent
        self.agent = self._initialize_agent()
        
        # 初始化内存
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2000
        )
        
        self.conversation_history = []
        
        logger.info("DevOps AI Agent初始化完成")
    
    def _validate_configuration(self):
        """验证配置"""
        if not self.config.validate_config():
            raise ConfigurationError("配置验证失败，请检查配置文件")
    
    def _initialize_tools(self) -> List[Tool]:
        """初始化工具列表"""
        tools = []
        
        try:
            # Docker操作工具
            tools.append(DockerOpsTool())
            
            # 系统监控工具
            tools.append(SystemMonitorTool())
            
            # 文件管理工具
            tools.append(FileManagerTool())
            
            # 日志分析工具
            tools.append(LogAnalyzerTool())
            
            # 性能分析工具
            tools.append(PerformanceAnalyzerTool())
            
            # 服务检查工具
            tools.append(ServiceCheckerTool())
            
            logger.info(f"成功初始化 {len(tools)} 个工具")
            return tools
            
        except Exception as e:
            logger.error(f"初始化工具失败: {e}")
            raise AIAgentError(f"工具初始化失败: {e}")
    
    def _initialize_agent(self):
        """初始化LangChain Agent"""
        try:
            # 系统提示模板
            system_prompt = SystemMessagePromptTemplate.from_template(
                """你是一个专业的DevOps AI助手，专门帮助用户管理DevOps任务。
                
功能包括：
- Docker容器和镜像管理
- 系统资源监控（CPU、内存、磁盘、网络）
- 文件和目录操作
- 日志文件分析和错误诊断
- Python程序性能分析
- 系统服务状态检查和重启

请使用自然语言与用户沟通，提供清晰、专业的建议。
当需要执行操作时，使用提供的工具。
如果不确定如何操作，请询问用户更多细节。

重要规则：
1. 始终优先考虑系统安全
2. 确认危险操作（如删除文件、重启服务）
3. 提供操作结果的详细解释
4. 如果工具执行失败，提供故障排除建议
5. 保持响应简洁但信息完整"""
            )
            
            # 初始化Agent
            agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=5,
                early_stopping_method="generate",
                memory=self.memory,
                return_intermediate_steps=True
            )
            
            logger.info("LangChain Agent初始化成功")
            return agent
            
        except Exception as e:
            logger.error(f"初始化Agent失败: {e}")
            raise AIAgentError(f"Agent初始化失败: {e}")
    
    def chat(self, user_input: str) -> Dict[str, Any]:
        """
        与Agent进行对话
        
        Args:
            user_input: 用户输入
            
        Returns:
            对话结果字典
        """
        try:
            logger.info(f"用户输入: {user_input}")
            
            # 添加到对话历史
            self.conversation_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            })
            
            # 执行Agent
            result = self.agent(
                user_input,
                callbacks=[self._create_callback()]
            )
            
            # 提取AI响应
            ai_response = result['output'] if isinstance(result, dict) else str(result)
            
            # 添加到对话历史
            self.conversation_history.append({
                'role': 'assistant',
                'content': ai_response,
                'timestamp': datetime.now().isoformat()
            })
            
            # 限制历史长度
            max_history = self.config.get('agent.max_conversation_history', 50)
            if len(self.conversation_history) > max_history:
                self.conversation_history = self.conversation_history[-max_history:]
            
            logger.info(f"AI响应: {ai_response}")
            
            return {
                'response': ai_response,
                'success': True,
                'intermediate_steps': result.get('intermediate_steps', []),
                'conversation_history': self.conversation_history.copy()
            }
            
        except Exception as e:
            logger.error(f"Agent对话执行失败: {e}")
            error_response = f"抱歉，我遇到了一个错误: {str(e)}。请稍后重试或提供更多细节。"
            
            self.conversation_history.append({
                'role': 'assistant',
                'content': error_response,
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'response': error_response,
                'success': False,
                'error': str(e),
                'conversation_history': self.conversation_history.copy()
            }
    
    def _create_callback(self) -> CallbackManagerForLLMRun:
        """创建回调管理器"""
        from langchain.callbacks.base import BaseCallbackHandler
        
        class DevOpsCallbackHandler(BaseCallbackHandler):
            def __init__(self, agent_instance):
                self.agent = agent_instance
                self.logger = get_logger("agent_callback")
            
            def on_llm_start(self, serialized, prompts, **kwargs):
                self.logger.debug(f"LLM开始处理: {prompts[0][:100]}...")
            
            def on_llm_end(self, response, **kwargs):
                self.logger.debug("LLM处理完成")
            
            def on_tool_start(self, serialized, input_str, **kwargs):
                self.logger.info(f"工具开始执行: {serialized['name']} - 输入: {input_str}")
            
            def on_tool_end(self, output, **kwargs):
                self.logger.info(f"工具执行完成: {output[:200]}...")
        
        return CallbackManagerForLLMRun([DevOpsCallbackHandler(self)])
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """清除对话历史"""
        self.conversation_history.clear()
        self.memory.clear()
        logger.info("对话历史已清除")
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        """获取可用工具信息"""
        tools_info = []
        for tool in self.tools:
            tools_info.append({
                'name': tool.name,
                'description': tool.description,
                'args_schema': str(tool.args_schema) if hasattr(tool, 'args_schema') else None
            })
        return tools_info
    
    def shutdown(self):
        """关闭Agent，清理资源"""
        try:
            self.memory.clear()
            logger.info("Agent已关闭")
        except Exception as e:
            logger.warning(f"关闭Agent时出现警告: {e}")


def create_devops_agent() -> DevOpsAgent:
    """
    创建DevOps Agent的工厂函数
    
    Returns:
        DevOpsAgent实例
    """
    return DevOpsAgent()


if __name__ == "__main__":
    # 测试Agent初始化
    try:
        agent = create_devops_agent()
        print("DevOps AI Agent初始化成功！")
        print(f"可用工具: {len(agent.tools)} 个")
        agent.shutdown()
    except Exception as e:
        print(f"初始化失败: {e}")
