"""
Configuration management module
Responsible for loading and managing application configuration
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class Settings:
    """Configuration management class"""
    
    def __init__(self, config_file: str = "config/config.local.yaml"):
        """
        Initialize configuration manager
        
        Args:
            config_file: Config file path
        """
        self.config_file = config_file
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        """Load config file"""
        try:
            # 加载环境变量
            load_dotenv()
            
            # 读取YAML配置文件
            config_path = Path(self.config_file)
            if not config_path.exists():
                logger.warning(f"Config file {self.config_file} does not exist, using default config")
                self._load_default_config()
                return
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            # 替换环境变量
            config_content = self._substitute_env_vars(config_content)
            
            # 解析YAML
            self.config = yaml.safe_load(config_content)
            
            logger.info(f"Successfully loaded config file: {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            self._load_default_config()
    
    def _substitute_env_vars(self, content: str) -> str:
        """Substitute environment variables in config file"""
        import re
        
        def replace_var(match):
            var_expr = match.group(1)
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
                return os.getenv(var_name, default_value)
            else:
                return os.getenv(var_expr, '')
        
        # 替换 ${VAR_NAME:default_value} 格式的环境变量
        pattern = r'\$\{([^}]+)\}'
        return re.sub(pattern, replace_var, content)
    
    def _load_default_config(self):
        """Load default config"""
        self.config = {
            'openai': {
                'api_key': os.getenv('OPENAI_API_KEY', ''),
                'model': os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
                'temperature': float(os.getenv('OPENAI_TEMPERATURE', '0.7')),
                'max_tokens': int(os.getenv('OPENAI_MAX_TOKENS', '2000')),
                'timeout': 30
            },
            'app': {
                'name': os.getenv('APP_NAME', 'AIOpsAgent'),
                'version': os.getenv('APP_VERSION', '1.0.0'),
                'debug': os.getenv('DEBUG', 'false').lower() == 'true',
                'timezone': os.getenv('TIMEZONE', 'Asia/Shanghai')
            },
            'logging': {
                'level': os.getenv('LOG_LEVEL', 'INFO'),
                'file': 'logs/agent.log',
                'max_size': '10MB',
                'backup_count': 5,
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'monitoring': {
                'cpu_threshold': float(os.getenv('CPU_THRESHOLD', '80.0')),
                'memory_threshold': float(os.getenv('MEMORY_THRESHOLD', '85.0')),
                'disk_threshold': float(os.getenv('DISK_THRESHOLD', '90.0')),
                'check_interval': int(os.getenv('MONITOR_INTERVAL', '30')),
                'enable_alerts': True
            },
            'docker': {
                'socket_path': os.getenv('DOCKER_SOCKET', 'unix://var/run/docker.sock'),
                'timeout': int(os.getenv('DOCKER_TIMEOUT', '30')),
                'api_version': 'auto'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value
        
        Args:
            key: Config key, supports dot-separated nested keys, e.g., 'openai.api_key'
            default: Default value
            
        Returns:
            Config value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set config value
        
        Args:
            key: Config key
            value: Config value
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI config"""
        return self.get('openai', {})
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get app config"""
        return self.get('app', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging config"""
        return self.get('logging', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring config"""
        return self.get('monitoring', {})
    
    def get_docker_config(self) -> Dict[str, Any]:
        """Get Docker config"""
        return self.get('docker', {})
    
    def validate_config(self) -> bool:
        """
        Validate config validity
        
        Returns:
            Whether config is valid
        """
        try:
            # 检查必需的配置项
            required_keys = [
                'openai.api_key',
                'app.name',
                'logging.level'
            ]
            
            for key in required_keys:
                value = self.get(key)
                if not value:
                    logger.error(f"Missing required config item: {key}")
                    return False
            
            # 检查OpenAI API密钥
            api_key = self.get('openai.api_key')
            if not api_key or api_key == 'your-openai-api-key-here':
                logger.error("Please set a valid OpenAI API key")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            return False
    
    def reload(self):
        """Reload config"""
        self._load_config()
        logger.info("Config reloaded successfully")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return full config dictionary"""
        return self.config.copy()


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """Get config instance"""
    return settings


def reload_settings():
    """Reload config"""
    global settings
    settings.reload()


# 便捷函数
def get_config(key: str, default: Any = None) -> Any:
    """Convenient function to get config value"""
    return settings.get(key, default)


def validate_config() -> bool:
    """Convenient function to validate config"""
    return settings.validate_config()
