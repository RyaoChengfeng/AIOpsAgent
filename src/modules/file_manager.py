"""
文件管理模块
提供文件和目录操作的工具函数和LangChain工具
"""

import os
import shutil
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from config.settings import get_config
from src.utils.logger import get_logger
from src.utils.helpers import get_file_info, create_backup_filename, validate_file_path, truncate_string, format_bytes
from src.utils.exceptions import FileOperationError, PermissionError
from datetime import datetime
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from config.settings import Settings

logger = get_logger(__name__)


class FileManagerConfig(BaseModel):
    """文件管理配置模型"""
    max_file_size: str = Field(default_factory=lambda: get_config('file_manager.max_file_size', '100MB'))
    allowed_extensions: List[str] = Field(default_factory=lambda: get_config('file_manager.allowed_extensions', ['.txt', '.log', '.conf', '.yaml', '.yml', '.json', '.py', '.sh']))
    search_depth: int = Field(default_factory=lambda: get_config('file_manager.search_depth', 10))


class FileAction(BaseModel):
    action: str = Field(description="Operation type: create_file, create_directory, delete_file, delete_directory, list_directory, search_files, read_file, backup_file, file_info")
    path: Optional[str] = Field(default=None, description="Target file or directory path")
    filename: Optional[str] = Field(default=None, description="Filename when applicable")
    content: Optional[str] = Field(default=None, description="File content when creating or overwriting")
    pattern: Optional[str] = Field(default=None, description="Search pattern (glob or keyword)")
    is_directory: Optional[bool] = Field(default=None, description="Whether target refers to a directory")


file_parser = PydanticOutputParser(pydantic_object=FileAction)

file_prompt = PromptTemplate(
    template="""You are a file operation parsing assistant. Extract structured fields from the user's natural language instruction.

Allowed actions (must exactly match one):
- create_file
- create_directory
- delete_file
- delete_directory
- list_directory
- search_files
- read_file
- backup_file
- file_info

Rules:
- path: absolute or relative path if present; otherwise None.
- filename: when creating a file without an explicit path.
- content: text following markers like 内容: or content:; otherwise None.
- pattern: glob like *.py or keyword like error; otherwise None.
- is_directory: true if explicitly a directory operation; else false.

Command: {command}

{format_instructions}""",
    input_variables=["command"],
    partial_variables={"format_instructions": file_parser.get_format_instructions()},
)


class FileManagerTool(BaseTool):
    """文件管理LangChain工具"""
    
    name: str = "file_manager"
    description: str = (
        "用于文件和目录操作的工具。支持创建、删除、修改文件，查询目录结构，"
        "搜索文件内容，备份文件等操作。"
        "输入应为具体的文件操作描述，如'在当前目录创建名为config.txt的文件'、"
        "'列出/home目录下的所有.py文件'或'搜索包含error的日志文件'"
    )
    args_schema: Optional[BaseModel] = None
    
    def _parse_command(self, text: str) -> FileAction:
        settings = Settings()
        openai_config = settings.get_openai_config()
        max_tokens = openai_config.get('max_tokens', 2000)
        model = openai_config.get('model', 'gpt-3.5-turbo')
        temperature = openai_config.get('temperature', 0)
        api_key = openai_config.get('api_key')
        base_url = openai_config.get('base_url')
        llm = ChatOpenAI(
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
        chain = file_prompt | llm | file_parser
        try:
            parsed = chain.invoke({"command": text})
            return parsed
        except Exception as e:
            logger.warning(f"AI file parsing failed: {e}")
            return FileAction(action="unknown")

    def _run(self, operation: str) -> str:
        """
        执行文件操作
        
        Args:
            operation: 文件操作描述
            
        Returns:
            操作结果
        """
        try:
            parsed = self._parse_command(operation)
            action = (parsed.action or "").strip()

            if action == "create_file":
                target = parsed.path or parsed.filename or ""
                return self._create_file(target, parsed.content)
            elif action == "create_directory":
                target = parsed.path or parsed.filename or ""
                return self._create_directory(target)
            elif action == "delete_file":
                target = parsed.path or parsed.filename or ""
                return self._delete(target, is_dir=False)
            elif action == "delete_directory":
                target = parsed.path or parsed.filename or ""
                return self._delete(target, is_dir=True)
            elif action == "list_directory":
                target = parsed.path or "."
                return self._list_directory(target)
            elif action == "search_files":
                pattern = parsed.pattern or "*"
                base = parsed.path or "."
                return self._search_files_ai(pattern, base)
            elif action == "read_file":
                target = parsed.path or parsed.filename or ""
                return self._read_file(target)
            elif action == "backup_file":
                target = parsed.path or parsed.filename or ""
                return self._backup_file(target)
            elif action == "file_info":
                target = parsed.path or parsed.filename or ""
                return self._get_file_info(target)

            operation_lower = ""
            
            if "创建文件" in operation_lower or "create file" in operation_lower:
                filename = self._extract_filename(operation)
                content = self._extract_content(operation)
                return self._create_file(filename, content)
            elif "创建目录" in operation_lower or "create directory" in operation_lower:
                dirname = self._extract_filename(operation)
                return self._create_directory(dirname)
            elif "删除" in operation_lower or "delete" in operation_lower:
                path = self._extract_path(operation)
                is_dir = "目录" in operation_lower or "directory" in operation_lower
                return self._delete(path, is_dir)
            elif "列出" in operation_lower or "list" in operation_lower:
                path = self._extract_path(operation)
                return self._list_directory(path)
            elif "搜索" in operation_lower or "search" in operation_lower:
                pattern = self._extract_search_pattern(operation)
                return self._search_files(pattern)
            elif "查看内容" in operation_lower or "read file" in operation_lower:
                path = self._extract_path(operation)
                return self._read_file(path)
            elif "备份" in operation_lower or "backup" in operation_lower:
                path = self._extract_path(operation)
                return self._backup_file(path)
            elif "文件信息" in operation_lower or "file info" in operation_lower:
                path = self._extract_path(operation)
                return self._get_file_info(path)
            else:
                return (
                    "支持的文件操作:\\n"
                    "- 创建文件/目录 (指定名称和内容)\\n"
                    "- 删除文件/目录 (指定路径)\\n"
                    "- 列出目录内容 (指定路径)\\n"
                    "- 搜索文件 (指定模式或关键词)\\n"
                    "- 查看文件内容 (指定路径)\\n"
                    "- 备份文件 (指定路径)\\n"
                    "- 获取文件信息 (指定路径)\\n"
                    "请提供更具体的操作描述。"
                )
                
        except Exception as e:
            logger.error(f"文件操作失败: {e}")
            raise FileOperationError(f"文件操作执行失败: {str(e)}")
    
    def _extract_filename(self, operation: str) -> str:
        parsed = self._parse_command(operation)
        if parsed and parsed.filename:
            return parsed.filename
        if '"' in operation:
            return operation.split('"')[1]
        if "'" in operation:
            return operation.split("'")[1]
        words = operation.split()
        return words[-1] if words else ""
    
    def _extract_content(self, operation: str) -> Optional[str]:
        parsed = self._parse_command(operation)
        if parsed and parsed.content:
            return parsed.content
        if "内容:" in operation:
            return operation.split("内容:", 1)[1].strip()
        op_lower = operation.lower()
        if "content:" in op_lower:
            return op_lower.split("content:", 1)[1].strip()
        return None
    
    def _extract_path(self, operation: str) -> str:
        parsed = self._parse_command(operation)
        if parsed and parsed.path:
            return parsed.path
        words = operation.split()
        return words[-1] if words else "."
    
    def _extract_search_pattern(self, operation: str) -> str:
        parsed = self._parse_command(operation)
        if parsed and parsed.pattern:
            return parsed.pattern
        keywords = ['error', 'warning', 'failed', 'exception']
        for keyword in keywords:
            if keyword in operation.lower():
                return f"*{keyword}*"
        return operation.split()[-1] if operation.split() else "*"
    
    def _create_file(self, filename: str, content: Optional[str] = None) -> str:
        """创建文件"""
        try:
            if not filename:
                return "请指定文件名。"
            
            path = Path(filename)
            if path.exists():
                return f"文件 '{filename}' 已存在。"
            config = FileManagerConfig()
            if path.suffix and config.allowed_extensions and path.suffix.lower() not in [ext.lower() for ext in config.allowed_extensions]:
                return f"不允许的文件扩展名 '{path.suffix}'。允许: {', '.join(config.allowed_extensions)}"
            
            if content is None:
                content = "# 新创建的文件内容"
            
            if not validate_file_path(str(path), must_exist=False):
                return f"父目录不存在，无法创建 '{filename}'。"
            path.write_text(content, encoding='utf-8')
            
            info = get_file_info(str(path))
            result = f"✅ 文件 '{filename}' 创建成功！\\n"
            result += f"大小: {info.get('size_formatted', '0B')}\\n"
            result += f"创建时间: {info.get('created', '未知')}\\n"
            result += f"路径: {path.absolute()}"
            
            return result
        except PermissionError:
            raise PermissionError(f"无权限创建文件 '{filename}'")
        except Exception as e:
            logger.error(f"创建文件失败: {e}")
            raise FileOperationError(f"无法创建文件 '{filename}': {str(e)}")
    
    def _create_directory(self, dirname: str) -> str:
        """创建目录"""
        try:
            if not dirname:
                return "请指定目录名。"
            
            path = Path(dirname)
            if path.exists():
                return f"目录 '{dirname}' 已存在。"
            
            path.mkdir(parents=True, exist_ok=True)
            result = f"✅ 目录 '{dirname}' 创建成功！\\n"
            result += f"路径: {path.absolute()}"
            
            return result
        except PermissionError:
            raise PermissionError(f"无权限创建目录 '{dirname}'")
        except Exception as e:
            logger.error(f"创建目录失败: {e}")
            raise FileOperationError(f"无法创建目录 '{dirname}': {str(e)}")
    
    def _delete(self, path: str, is_dir: bool = False) -> str:
        """删除文件或目录"""
        try:
            if not path:
                return "请指定要删除的路径。"
            
            full_path = Path(path)
            if not full_path.exists():
                return f"路径 '{path}' 不存在。"
            
            if is_dir:
                if full_path.is_dir():
                    shutil.rmtree(full_path)
                    return f"✅ 目录 '{path}' 已删除。"
                else:
                    return f"'{path}' 不是目录。"
            else:
                if full_path.is_file():
                    full_path.unlink()
                    return f"✅ 文件 '{path}' 已删除。"
                else:
                    return f"'{path}' 不是文件。"
        except PermissionError:
            raise PermissionError(f"无权限删除 '{path}'")
        except Exception as e:
            logger.error(f"删除操作失败: {e}")
            raise FileOperationError(f"无法删除 '{path}': {str(e)}")
    
    def _list_directory(self, path: str = ".") -> str:
        """列出目录内容"""
        try:
            full_path = Path(path)
            if not full_path.exists():
                return f"目录 '{path}' 不存在。"
            
            if not full_path.is_dir():
                return f"'{path}' 不是目录。"
            
            items = list(full_path.iterdir())
            if not items:
                return f"目录 '{path}' 是空的。"
            
            result = f"目录 '{path}' 内容 ({len(items)} 个项目):\\n"
            result += "-" * 50 + "\\n"
            
            for item in items:
                info = get_file_info(str(item))
                prefix = "📁 " if item.is_dir() else "📄 "
                size = info.get('size_formatted', '-') if item.is_file() else '-'
                result += f"{prefix}{item.name}\\t{size}\\n"
            
            return result
        except PermissionError:
            raise PermissionError(f"无权限访问目录 '{path}'")
        except Exception as e:
            logger.error(f"列出目录失败: {e}")
            raise FileOperationError(f"无法列出目录 '{path}': {str(e)}")
    
    def _search_files(self, pattern: str) -> str:
        """搜索文件"""
        try:
            if not pattern:
                pattern = "*"
            
            current_dir = Path(".")
            config = FileManagerConfig()
            matches = [p for p in current_dir.rglob(pattern)]
            filtered = []
            for p in matches:
                try:
                    depth = len(p.relative_to(current_dir).parts)
                except ValueError:
                    depth = 0
                if depth <= config.search_depth:
                    filtered.append(p)
            matches = filtered
            
            if not matches:
                return f"未找到匹配 '{pattern}' 的文件。"
            
            result = f"搜索结果 (匹配 '{pattern}'): ({len(matches)} 个文件)\\n"
            result += "-" * 40 + "\\n"
            
            for match in matches:
                info = get_file_info(str(match))
                size = info.get('size_formatted', '0B')
                modified = info.get('modified', '未知').strftime('%Y-%m-%d %H:%M') if info.get('modified') else '未知'
                result += f"{match.name}\\t{size}\\t修改: {modified}\\n"
            
            return result
        except Exception as e:
            logger.error(f"搜索文件失败: {e}")
            raise FileOperationError(f"搜索失败: {str(e)}")

    def _search_files_ai(self, pattern: str, base: Optional[str] = None) -> str:
        """基于 AI 解析的搜索（支持指定基目录）"""
        try:
            if not pattern:
                pattern = "*"
            current_dir = Path(base or ".")
            config = FileManagerConfig()
            matches = [p for p in current_dir.rglob(pattern)]
            filtered: List[Path] = []
            for p in matches:
                try:
                    depth = len(p.relative_to(current_dir).parts)
                except ValueError:
                    depth = 0
                if depth <= config.search_depth:
                    filtered.append(p)
            if not filtered:
                return f"未找到匹配 '{pattern}' 的文件。"
            result = f"搜索结果 (匹配 '{pattern}'): ({len(filtered)} 个文件)\n"
            result += "-" * 40 + "\n"
            for match in filtered:
                info = get_file_info(str(match))
                size = info.get('size_formatted', '0B')
                modified = info.get('modified', '未知')
                modified_str = modified.strftime('%Y-%m-%d %H:%M') if hasattr(modified, 'strftime') else '未知'
                result += f"{match.name}\t{size}\t修改: {modified_str}\n"
            return result
        except Exception as e:
            logger.error(f"搜索文件失败: {e}")
            raise FileOperationError(f"搜索失败: {str(e)}")
    
    def _read_file(self, path: str) -> str:
        """读取文件内容"""
        try:
            full_path = Path(path)
            if not full_path.exists():
                return f"文件 '{path}' 不存在。"
            
            if not full_path.is_file():
                return f"'{path}' 不是文件。"
            
            info = get_file_info(str(full_path))
            size_bytes = info.get('size', 0)
            
            config = FileManagerConfig()
            if full_path.suffix and config.allowed_extensions and full_path.suffix.lower() not in [ext.lower() for ext in config.allowed_extensions]:
                return f"不允许读取扩展名为 '{full_path.suffix}' 的文件。允许: {', '.join(config.allowed_extensions)}"
            max_size_bytes = self._parse_size(config.max_file_size)
            
            if size_bytes > max_size_bytes:
                return f"文件 '{path}' 太大 ({format_bytes(size_bytes)} > {config.max_file_size})，无法读取完整内容。"
            
            content = full_path.read_text(encoding='utf-8', errors='ignore')
            
            # 截断长内容
            if len(content) > 5000:
                content = content[:5000] + "\\n... (内容已截断，完整文件大小: " + format_bytes(size_bytes) + ")"
            
            result = f"文件 '{path}' 内容 (大小: {format_bytes(size_bytes)}):\\n"
            result += "-" * 50 + "\\n"
            result += content
            
            return result
        except PermissionError:
            raise PermissionError(f"无权限读取文件 '{path}'")
        except UnicodeDecodeError:
            return f"无法读取文件 '{path}' (编码问题)。尝试使用二进制模式。"
        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            raise FileOperationError(f"无法读取文件 '{path}': {str(e)}")
    
    def _backup_file(self, path: str) -> str:
        """备份文件"""
        try:
            full_path = Path(path)
            if not full_path.exists():
                return f"文件 '{path}' 不存在，无法备份。"
            
            if not full_path.is_file():
                return f"'{path}' 不是文件。"
            
            backup_path = create_backup_filename(str(full_path))
            shutil.copy2(full_path, backup_path)
            
            result = f"✅ 文件 '{path}' 已备份到 '{backup_path}'。\\n"
            result += f"备份时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return result
        except PermissionError:
            raise PermissionError(f"无权限备份文件 '{path}'")
        except Exception as e:
            logger.error(f"备份文件失败: {e}")
            raise FileOperationError(f"无法备份文件 '{path}': {str(e)}")
    
    def _get_file_info(self, path: str) -> str:
        """获取文件信息"""
        try:
            full_path = Path(path)
            if not full_path.exists():
                return f"路径 '{path}' 不存在。"
            
            info = get_file_info(str(full_path))
            
            result = f"文件/目录 '{path}' 信息:\\n"
            result += "-" * 30 + "\\n"
            result += f"类型: {'目录' if info.get('is_dir') else '文件'}\\n"
            result += f"名称: {info.get('name', '未知')}\\n"
            result += f"大小: {info.get('size_formatted', '0B')}\\n"
            result += f"修改时间: {info.get('modified', '未知')}\\n"
            result += f"创建时间: {info.get('created', '未知')}\\n"
            result += f"权限: {info.get('permissions', '未知')}\\n"
            result += f"绝对路径: {full_path.absolute()}"
            
            return result
        except Exception as e:
            logger.error(f"获取文件信息失败: {e}")
            raise FileOperationError(f"无法获取 '{path}' 信息: {str(e)}")
    
    def _parse_size(self, size_str: str) -> int:
        """解析文件大小字符串"""
        size_str = size_str.upper().strip()
        if size_str.endswith('KB'):
            return int(float(size_str[:-2]) * 1024)
        elif size_str.endswith('MB'):
            return int(float(size_str[:-2]) * 1024 * 1024)
        elif size_str.endswith('GB'):
            return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
        else:
            return int(size_str)


if __name__ == "__main__":
    # 测试文件管理工具
    try:
        tool = FileManagerTool()
        print("测试文件管理工具:")
        print(tool._run("列出 ."))
    except Exception as e:
        print(f"测试失败: {e}")
