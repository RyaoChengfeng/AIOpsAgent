"""
Core logic module for the AI Agent.
Implements a LangChain-based intelligent DevOps Agent.
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
from langchain.agents import initialize_agent, AgentType
# from langchain.llms import OpenAI
# from langchain_openai import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks.manager import CallbackManagerForLLMRun
from config.settings import Settings
from src.utils.logger import get_logger
from src.utils.exceptions import AIAgentError, ConfigurationError
from src.modules import DockerOpsTool, SystemMonitorTool, FileManagerTool, LogAnalyzerTool, ServiceCheckerTool

logger = get_logger(__name__)


class DevOpsAgent:
    """Main class for the DevOps AI Agent."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the DevOps AI Agent.

        Args:
            config_path: Path to the configuration file.
        """
        self.config = Settings()
        self._validate_configuration()

        # Load OpenAI configuration
        openai_config = self.config.get_openai_config()
        api_key = openai_config.get('api_key')
        base_url = openai_config.get('base_url')
        model = openai_config.get('model', 'gpt-3.5-turbo')
        temperature = openai_config.get('temperature', 0.7)
        max_tokens = openai_config.get('max_tokens', 2000)

        if not api_key or api_key == 'your-openai-api-key-here':
            raise ConfigurationError("A valid OpenAI API key must be provided.")

        os.environ['OPENAI_API_KEY'] = api_key

        # Initialize language model
        try:
            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                openai_api_key=api_key,
                openai_api_base=base_url,
                max_tokens=max_tokens,
                default_headers={
                    "HTTP-Referer": "https://localhost/",
                    "X-Title": "DevOps-AIOps-Agent"
                }
            )
            logger.info(f"LLM initialized successfully: {model} (base_url: {base_url})")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise AIAgentError(f"Unable to initialize AI model: {e}")

        # Initialize tools
        self.tools = self._initialize_tools()

        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            # max_token_limit removed
        )

        # Initialize conversation history log
        self.conversation_history: List[Dict[str, Any]] = []

        # Initialize Agent
        self.agent = self._initialize_agent()
        logger.info("DevOps AI Agent initialized successfully.")

    def _validate_configuration(self):
        """Validate configuration."""
        if not self.config.validate_config():
            raise ConfigurationError("Configuration validation failed. Please check the config file.")

    def _initialize_tools(self) -> List[Tool]:
        """Initialize the list of tools."""
        tools = []

        try:
            tools.append(DockerOpsTool())
            tools.append(SystemMonitorTool())
            tools.append(FileManagerTool())
            tools.append(LogAnalyzerTool())

            # PerformanceAnalyzerTool could be added here
            # tools.append(PerformanceAnalyzerTool())

            tools.append(ServiceCheckerTool())

            logger.info(f"{len(tools)} tools initialized successfully.")
            return tools

        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            raise AIAgentError(f"Tool initialization failed: {e}")

    def _initialize_agent(self):
        """Initialize the LangChain Agent."""
        try:
            # System prompt template
            system_prompt = SystemMessagePromptTemplate.from_template(
                """
                You are a professional DevOps AI assistant that helps users manage DevOps tasks.
                Capabilities include:
                - Docker container and image management
                - System resource monitoring (CPU, memory, disk, network)
                - File and directory operations
                - Log file analysis and error diagnostics
                - Python program performance analysis
                - System service status checks and restarts

                Communicate naturally with the user and provide clear, professional advice.
                When actions are needed, use the provided tools.
                If uncertain, request additional details from the user.

                Important Rules:
                1. Always prioritize system safety.
                2. Confirm dangerous operations (e.g., deleting files, restarting services).
                3. Provide detailed explanations of results.
                4. If a tool fails, provide troubleshooting suggestions.
                5. Keep responses concise but complete."""
            )

            agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=5,
                early_stopping_method="generate",
                memory=self.memory,
                return_intermediate_steps=False
            )

            logger.info("LangChain Agent initialized successfully.")
            return agent

        except Exception as e:
            logger.error(f"Failed to initialize Agent: {e}")
            raise AIAgentError(f"Agent initialization failed: {e}")

    def chat(self, user_input: str) -> Dict[str, Any]:
        """
        Conduct chat interaction with the Agent.
        Args:
            user_input: User query.
        Returns:
            Response dictionary.
        """
        try:
            logger.info(f"User input: {user_input}")

            self.conversation_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            })

            result = self.agent.invoke(
                {"input": user_input},
                config={"callbacks": self._create_callback()}
            )

            if isinstance(result, dict):
                ai_response = result.get('output', '')
                intermediate_steps = result.get('intermediate_steps', [])
            else:
                ai_response = str(result)
                intermediate_steps = []

            self.conversation_history.append({
                'role': 'assistant',
                'content': ai_response,
                'intermediate_steps': intermediate_steps,
                'timestamp': datetime.now().isoformat()
            })

            max_history = self.config.get('agent.max_conversation_history', 50)
            if len(self.conversation_history) > max_history:
                self.conversation_history = self.conversation_history[-max_history:]

            logger.info(f"AI response: {ai_response}")

            return {
                'response': ai_response,
                'success': True,
                'intermediate_steps': result.get('intermediate_steps', []),
                'conversation_history': self.conversation_history.copy()
            }

        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            error_response = f"Sorry, an error occurred: {str(e)}. Please try again later or provide more details."

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
        """Create callback manager."""
        from langchain.callbacks.base import BaseCallbackHandler

        class DevOpsCallbackHandler(BaseCallbackHandler):
            def __init__(self, agent_instance):
                self.agent = agent_instance
                self.logger = get_logger("agent_callback")

            def on_llm_start(self, serialized, prompts, **kwargs):
                self.logger.debug(f"LLM start: {prompts[0][:100]}...")

            def on_llm_end(self, response, **kwargs):
                self.logger.debug("LLM processing complete.")

            def on_tool_start(self, serialized, input_str, **kwargs):
                self.logger.info(f"Tool start: {serialized['name']} - Input: {input_str}")

            def on_tool_end(self, output, **kwargs):
                self.logger.info(f"Tool finished: {output[:200]}...")

        return CallbackManagerForLLMRun(
            run_id=uuid.uuid4(),
            handlers=[],
            inheritable_handlers=[DevOpsCallbackHandler(self)]
        )

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Return conversation history."""
        return self.conversation_history.copy()

    def clear_conversation_history(self):
        """Clear conversation history and memory."""
        self.conversation_history.clear()
        self.memory.clear()
        logger.info("Conversation history cleared.")

    def get_available_tools(self) -> List[Dict[str, str]]:
        """Return metadata about available tools."""
        tools_info = []
        for tool in self.tools:
            tools_info.append({
                'name': tool.name,
                'description': tool.description,
                'args_schema': str(tool.args_schema) if hasattr(tool, 'args_schema') else None
            })
        return tools_info

    def shutdown(self):
        """Shutdown the Agent and clean resources."""
        try:
            self.memory.clear()
            logger.info("Agent shutdown completed.")
        except Exception as e:
            logger.warning(f"Warning during shutdown: {e}")


def create_devops_agent() -> DevOpsAgent:
    """
    Factory function to create DevOps Agent.

    Returns:
        DevOpsAgent instance.
    """
    return DevOpsAgent()


if __name__ == "__main__":
    try:
        agent = create_devops_agent()
        print("DevOps AI Agent initialized successfully!")
        print(f"Available tools: {len(agent.tools)}")
        agent.shutdown()
    except Exception as e:
        print(f"Initialization failed: {e}")
