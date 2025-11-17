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
from src.utils.helpers import get_file_info, create_backup_filename, validate_file_path, truncate_string
from src.utils.exceptions import FileOperationError, PermissionError

logger = get_logger(__name__)


class FileManagerConfig(BaseModel):
    """文件管理配置模型"""
    max_file_size: str = Field(default_factory=lambda: get_config('file_manager.max_file_size', '100MB'))
    allowed_extensions: List[str] = Field(default_factory=lambda: get_config('file_manager.allowed_extensions', ['.txt', '.log', '.conf', '.yaml', '.yml', '.json', '.py', '.sh']))
    search_depth: int = Field(default_factory=lambda: get_config('file_manager.search_depth', 10))


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
    
    def _run(self, operation: str) -> str:
        """
        执行文件操作
        
        Args:
            operation: 文件操作描述
            
        Returns:
            操作结果
        """
        try:
            operation_lower = operation.lower()
            
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
        """从操作描述中提取文件名"""
        # 简单提取引号内或最后一个词作为文件名
        if '"' in operation:
            return operation.split('"')[1]
        words = operation.split()
        return words[-1] if words else ""
    
    def _extract_content(self, operation: str) -> Optional[str]:
        """从操作描述中提取文件内容"""
        if "内容" in operation or "content" in operation:
            # 假设内容在操作描述的最后部分
            parts = operation.split("内容:", 1)
            if len(parts) > 1:
                return parts[1].strip()
        return None
    
    def _extract_path(self, operation: str) -> str:
        """从操作描述中提取路径"""
        # 简单提取最后一个词作为路径
        words = operation.split()
        return words[-1] if words else "."
    
    def _extract_search_pattern(self, operation: str) -> str:
        """从操作描述中提取搜索模式"""
        # 提取操作描述中的关键词
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
            
            if content is None:
                content = "# 新创建的文件内容"
            
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
            
            # 在当前目录搜索
            current_dir = Path(".")
            matches = list(current_dir.glob(pattern, recursive=False))
            
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
        print(tool._run("列出当前目录"))
    except Exception as e:
        print(f"测试失败: {e}")
