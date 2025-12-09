"""
File Management Module
Provides utility functions and LangChain tools for file and directory operations
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
    """File Management Configuration Model"""
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
- content: text following markers like '内容:' or 'content:'; otherwise None.
- pattern: glob like *.py or keyword like error; otherwise None.
- is_directory: true if explicitly a directory operation; else false.

Command: {command}

{format_instructions}""",
    input_variables=["command"],
    partial_variables={"format_instructions": file_parser.get_format_instructions()},
)


class FileManagerTool(BaseTool):
    """File Management LangChain Tool"""

    name: str = "file_manager"
    description: str = (
        "A tool for file and directory operations. Supports creating, deleting, modifying files, querying directory structure, "
        "searching file content, backing up files, and other operations."
        "The input should be a concrete file operation description, such as 'create a file named config.txt in the current directory', "
        "'list all .py files under the /home directory', or 'search log files containing error'"
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
        Executes the file operation

        Args:
            operation: Description of the file operation

        Returns:
            The result of the operation
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
            else:
                return (
                    "Supported file operations:\\n"
                    "- Create file/directory (specify name and content)\\n"
                    "- Delete file/directory (specify path)\\n"
                    "- List directory content (specify path)\\n"
                    "- Search files (specify pattern or keyword)\\n"
                    "- Read file content (specify path)\\n"
                    "- Backup file (specify path)\\n"
                    "- Get file info (specify path)\\n"
                    "Please provide a more specific operation description."
                )

        except Exception as e:
            logger.error(f"File operation failed: {e}")
            raise FileOperationError(f"File operation execution failed: {str(e)}")

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
        """Create file"""
        try:
            if not filename:
                return "Please specify a filename."

            path = Path(filename)
            if path.exists():
                return f"File '{filename}' already exists."
            config = FileManagerConfig()
            if path.suffix and config.allowed_extensions and path.suffix.lower() not in [ext.lower() for ext in config.allowed_extensions]:
                return f"Disallowed file extension '{path.suffix}'. Allowed: {', '.join(config.allowed_extensions)}"

            if content is None:
                content = "# New created file content"

            if not validate_file_path(str(path), must_exist=False):
                return f"Parent directory does not exist, cannot create '{filename}'."
            path.write_text(content, encoding='utf-8')

            info = get_file_info(str(path))
            result = f"✅ File '{filename}' created successfully!\\n"
            result += f"Size: {info.get('size_formatted', '0B')}\\n"
            result += f"Created Time: {info.get('created', 'Unknown')}\\n"
            result += f"Path: {path.absolute()}"

            return result
        except PermissionError:
            raise PermissionError(f"Permission denied to create file '{filename}'")
        except Exception as e:
            logger.error(f"Failed to create file: {e}")
            raise FileOperationError(f"Could not create file '{filename}': {str(e)}")

    def _create_directory(self, dirname: str) -> str:
        """Create directory"""
        try:
            if not dirname:
                return "Please specify a directory name."

            path = Path(dirname)
            if path.exists():
                return f"Directory '{dirname}' already exists."

            path.mkdir(parents=True, exist_ok=True)
            result = f"✅ Directory '{dirname}' created successfully!\\n"
            result += f"Path: {path.absolute()}"

            return result
        except PermissionError:
            raise PermissionError(f"Permission denied to create directory '{dirname}'")
        except Exception as e:
            logger.error(f"Failed to create directory: {e}")
            raise FileOperationError(f"Could not create directory '{dirname}': {str(e)}")

    def _delete(self, path: str, is_dir: bool = False) -> str:
        """Delete file or directory"""
        try:
            if not path:
                return "Please specify the path to delete."

            full_path = Path(path)
            if not full_path.exists():
                return f"Path '{path}' does not exist."

            if is_dir:
                if full_path.is_dir():
                    shutil.rmtree(full_path)
                    return f"✅ Directory '{path}' deleted."
                else:
                    return f"'{path}' is not a directory."
            else:
                if full_path.is_file():
                    full_path.unlink()
                    return f"✅ File '{path}' deleted."
                else:
                    return f"'{path}' is not a file."
        except PermissionError:
            raise PermissionError(f"Permission denied to delete '{path}'")
        except Exception as e:
            logger.error(f"Delete operation failed: {e}")
            raise FileOperationError(f"Could not delete '{path}': {str(e)}")

    def _list_directory(self, path: str = ".") -> str:
        """List directory content"""
        try:
            full_path = Path(path)
            if not full_path.exists():
                return f"Directory '{path}' does not exist."

            if not full_path.is_dir():
                return f"'{path}' is not a directory."

            items = list(full_path.iterdir())
            if not items:
                return f"Directory '{path}' is empty."

            result = f"Content of directory '{path}' ({len(items)} items):\\n"
            result += "-" * 50 + "\\n"

            for item in items:
                info = get_file_info(str(item))
                prefix = "📁 " if item.is_dir() else "📄 "
                size = info.get('size_formatted', '-') if item.is_file() else '-'
                result += f"{prefix}{item.name}\\t{size}\\n"

            return result
        except PermissionError:
            raise PermissionError(f"Permission denied to access directory '{path}'")
        except Exception as e:
            logger.error(f"Failed to list directory: {e}")
            raise FileOperationError(f"Could not list directory '{path}': {str(e)}")

    def _search_files(self, pattern: str) -> str:
        """Search files"""
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
                return f"No files matching '{pattern}' found."

            result = f"Search results (matching '{pattern}'): ({len(matches)} files)\\n"
            result += "-" * 40 + "\\n"

            for match in matches:
                info = get_file_info(str(match))
                size = info.get('size_formatted', '0B')
                modified = info.get('modified', 'Unknown').strftime('%Y-%m-%d %H:%M') if info.get('modified') else 'Unknown'
                result += f"{match.name}\\t{size}\\tModified: {modified}\\n"

            return result
        except Exception as e:
            logger.error(f"Failed to search files: {e}")
            raise FileOperationError(f"Search failed: {str(e)}")

    def _search_files_ai(self, pattern: str, base: Optional[str] = None) -> str:
        """AI-parsed search (supports specifying base directory)"""
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
                return f"No files matching '{pattern}' found."
            result = f"Search results (matching '{pattern}'): ({len(filtered)} files)\n"
            result += "-" * 40 + "\n"
            for match in filtered:
                info = get_file_info(str(match))
                size = info.get('size_formatted', '0B')
                modified = info.get('modified', 'Unknown')
                modified_str = modified.strftime('%Y-%m-%d %H:%M') if hasattr(modified, 'strftime') else 'Unknown'
                result += f"{match.name}\t{size}\tModified: {modified_str}\n"
            return result
        except Exception as e:
            logger.error(f"Failed to search files: {e}")
            raise FileOperationError(f"Search failed: {str(e)}")

    def _read_file(self, path: str) -> str:
        """Read file content"""
        try:
            full_path = Path(path)
            if not full_path.exists():
                return f"File '{path}' does not exist."

            if not full_path.is_file():
                return f"'{path}' is not a file."

            info = get_file_info(str(full_path))
            size_bytes = info.get('size', 0)

            config = FileManagerConfig()
            if full_path.suffix and config.allowed_extensions and full_path.suffix.lower() not in [ext.lower() for ext in config.allowed_extensions]:
                return f"Reading file with extension '{full_path.suffix}' is not allowed. Allowed: {', '.join(config.allowed_extensions)}"
            max_size_bytes = self._parse_size(config.max_file_size)

            if size_bytes > max_size_bytes:
                return f"File '{path}' is too large ({format_bytes(size_bytes)} > {config.max_file_size}) to read full content."

            content = full_path.read_text(encoding='utf-8', errors='ignore')

            # Truncate long content
            if len(content) > 5000:
                content = content[:5000] + "\\n... (Content truncated, full file size: " + format_bytes(size_bytes) + ")"

            result = f"Content of file '{path}' (Size: {format_bytes(size_bytes)}):\\n"
            result += "-" * 50 + "\\n"
            result += content

            return result
        except PermissionError:
            raise PermissionError(f"Permission denied to read file '{path}'")
        except UnicodeDecodeError:
            return f"Could not read file '{path}' (encoding issue). Try using binary mode."
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            raise FileOperationError(f"Could not read file '{path}': {str(e)}")

    def _backup_file(self, path: str) -> str:
        """Backup file"""
        try:
            full_path = Path(path)
            if not full_path.exists():
                return f"File '{path}' does not exist, cannot backup."

            if not full_path.is_file():
                return f"'{path}' is not a file."

            backup_path = create_backup_filename(str(full_path))
            shutil.copy2(full_path, backup_path)

            result = f"✅ File '{path}' backed up to '{backup_path}'.\\n"
            result += f"Backup Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            return result
        except PermissionError:
            raise PermissionError(f"Permission denied to backup file '{path}'")
        except Exception as e:
            logger.error(f"Failed to backup file: {e}")
            raise FileOperationError(f"Could not backup file '{path}': {str(e)}")

    def _get_file_info(self, path: str) -> str:
        """Get file information"""
        try:
            full_path = Path(path)
            if not full_path.exists():
                return f"Path '{path}' does not exist."

            info = get_file_info(str(full_path))

            result = f"File/Directory '{path}' Info:\\n"
            result += "-" * 30 + "\\n"
            result += f"Type: {'Directory' if info.get('is_dir') else 'File'}\\n"
            result += f"Name: {info.get('name', 'Unknown')}\\n"
            result += f"Size: {info.get('size_formatted', '0B')}\\n"
            result += f"Modified Time: {info.get('modified', 'Unknown')}\\n"
            result += f"Created Time: {info.get('created', 'Unknown')}\\n"
            result += f"Permissions: {info.get('permissions', 'Unknown')}\\n"
            result += f"Absolute Path: {full_path.absolute()}"

            return result
        except Exception as e:
            logger.error(f"Failed to get file information: {e}")
            raise FileOperationError(f"Could not get info for '{path}': {str(e)}")

    def _parse_size(self, size_str: str) -> int:
        """Parse file size string"""
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
    # Test file management tool
    try:
        tool = FileManagerTool()
        print("Testing file management tool:")
        print(tool._run("list ."))
    except Exception as e:
        print(f"Test failed: {e}")