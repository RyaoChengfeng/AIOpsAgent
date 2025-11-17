"""
日志分析模块
提供日志文件解析和分析的工具函数和LangChain工具
"""

import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from config.settings import get_config
from src.utils.logger import get_logger
from src.utils.helpers import parse_log_level, extract_error_patterns, safe_json_loads, truncate_string, run_command
from src.utils.exceptions import LogAnalysisError, FileOperationError
from pathlib import Path

logger = get_logger(__name__)


class LogAnalyzerConfig(BaseModel):
    """日志分析配置模型"""
    max_file_size: str = Field(default_factory=lambda: get_config('file_manager.max_file_size', '100MB'))
    error_patterns: List[str] = Field(default_factory=lambda: [
        'error', 'exception', 'failed', 'timeout', 'connection refused', 'permission denied'
    ])
    warning_patterns: List[str] = Field(default_factory=lambda: [
        'warning', 'deprecated', 'notice'
    ])


class LogAnalyzerTool(BaseTool):
    """日志分析LangChain工具"""
    
    name: str = "log_analyzer"
    description: str = (
        "用于分析日志文件的工具。支持解析服务日志，搜索错误或异常，"
        "自动分类日志内容并生成报告，统计日志级别等操作。"
        "输入应为具体的日志分析请求，如'分析/var/log/nginx/error.log中的错误'、"
        "'生成access.log的访问统计报告'或'搜索所有日志中的数据库连接错误'"
    )
    args_schema: Optional[BaseModel] = None
    
    def _run(self, analysis_request: str) -> str:
        """
        执行日志分析操作
        
        Args:
            analysis_request: 日志分析请求描述
            
        Returns:
            分析结果
        """
        try:
            request_lower = analysis_request.lower()
            
            if "分析" in request_lower or "analyze" in request_lower:
                log_file = self._extract_log_file(analysis_request)
                if log_file:
                    if "错误" in request_lower or "error" in request_lower:
                        return self._analyze_errors(log_file)
                    elif "警告" in request_lower or "warning" in request_lower:
                        return self._analyze_warnings(log_file)
                    elif "报告" in request_lower or "report" in request_lower:
                        return self._generate_report(log_file)
                    else:
                        return self._analyze_log_file(log_file)
                else:
                    return "请指定日志文件路径。"
            elif "搜索" in request_lower or "search" in request_lower:
                keyword = self._extract_keyword(analysis_request)
                log_file = self._extract_log_file(analysis_request)
                if log_file and keyword:
                    return self._search_log(log_file, keyword)
                elif keyword:
                    return self._search_all_logs(keyword)
                else:
                    return "请指定搜索关键词和日志文件。"
            elif "统计" in request_lower or "statistics" in request_lower:
                log_file = self._extract_log_file(analysis_request)
                if log_file:
                    return self._get_log_statistics(log_file)
                else:
                    return "请指定日志文件。"
            else:
                return (
                    "支持的日志分析操作:\\n"
                    "- 分析日志文件 (指定路径)\\n"
                    "- 搜索日志中的特定关键词 (指定文件和关键词)\\n"
                    "- 生成日志报告 (指定文件)\\n"
                    "- 统计日志级别分布 (指定文件)\\n"
                    "- 分析错误/警告日志 (指定文件)\\n"
                    "示例: '分析/var/log/nginx/error.log中的错误'"
                )
                
        except Exception as e:
            logger.error(f"日志分析失败: {e}")
            raise LogAnalysisError(f"日志分析执行失败: {str(e)}")
    
    def _extract_log_file(self, request: str) -> Optional[str]:
        """从请求中提取日志文件路径"""
        # 常见日志路径
        common_logs = [
            '/var/log/nginx/error.log', '/var/log/nginx/access.log',
            '/var/log/apache2/error.log', '/var/log/apache2/access.log',
            '/var/log/mysql/error.log', '/var/log/syslog',
            '/var/log/auth.log', '/var/log/kern.log'
        ]
        
        for log_path in common_logs:
            if log_path in request:
                return log_path
        
        # 提取路径（最后一个词或引号内）
        if '"' in request:
            return request.split('"')[1]
        words = request.split()
        if len(words) > 2:
            return words[-1]
        
        return None
    
    def _extract_keyword(self, request: str) -> Optional[str]:
        """从请求中提取关键词"""
        keywords = ['error', 'warning', 'failed', 'timeout', 'connection', 'permission']
        for keyword in keywords:
            if keyword in request.lower():
                return keyword
        
        # 提取操作描述中的关键词
        words = request.split()
        for word in words:
            if len(word) > 3 and word.lower() not in ['分析', 'analyze', '日志', 'log', '文件', 'file']:
                return word.lower()
        
        return None
    
    def _analyze_log_file(self, log_file: str) -> str:
        """分析日志文件"""
        try:
            if not self._validate_log_file(log_file):
                return f"无法访问日志文件 '{log_file}' 或文件太大。"
            
            content = self._read_log_file(log_file)
            if not content:
                return f"日志文件 '{log_file}' 为空或无法读取。"
            
            # 基本统计
            stats = parse_log_level(content)
            errors = extract_error_patterns(content)
            
            config = LogAnalyzerConfig()
            max_size_bytes = self._parse_size(config.max_file_size)
            
            result = f"日志文件 '{log_file}' 分析报告:\\n"
            result += "=" * 50 + "\n"
            result += f"文件大小: {self._get_file_size(log_file)}\n"
            result += f"行数: {len(content.splitlines())}\n\n"
            
            result += "日志级别统计:\n"
            result += "-" * 20 + "\n"
            for level, count in stats.items():
                percentage = (count / sum(stats.values()) * 100) if sum(stats.values()) > 0 else 0
                result += f"{level}: {count} ({percentage:.1f}%)\n"
            
            if errors:
                result += f"\n发现 {len(errors)} 个错误模式:\n"
                result += "-" * 20 + "\n"
                for error in errors[:10]:  # 显示前10个
                    result += f"- {truncate_string(error, 80)}\n"
                if len(errors) > 10:
                    result += f"\n... 还有 {len(errors) - 10} 个错误\n"
            else:
                result += "\n✅ 未发现明显的错误模式。"
            
            # 最近10行日志
            lines = content.splitlines()
            recent_logs = '\n'.join(lines[-10:])
            result += f"\n最近10行日志:\n"
            result += "-" * 20 + "\n"
            result += truncate_string(recent_logs, 1000)
            
            return result
            
        except Exception as e:
            logger.error(f"分析日志文件失败: {e}")
            raise LogAnalysisError(f"无法分析日志文件 '{log_file}': {str(e)}")
    
    def _analyze_errors(self, log_file: str) -> str:
        """分析错误日志"""
        try:
            if not self._validate_log_file(log_file):
                return f"无法访问日志文件 '{log_file}'。"
            
            content = self._read_log_file(log_file)
            errors = extract_error_patterns(content)
            
            if not errors:
                return f"✅ 在日志文件 '{log_file}' 中未发现错误。"
            
            # 分类错误
            error_types = {
                '连接错误': 0,
                '权限错误': 0,
                '超时错误': 0,
                '文件错误': 0,
                '其他错误': 0
            }
            
            for error in errors:
                error_lower = error.lower()
                if any(keyword in error_lower for keyword in ['connection', 'refused', 'timeout']):
                    error_types['连接错误'] += 1
                elif any(keyword in error_lower for keyword in ['permission', 'access denied']):
                    error_types['权限错误'] += 1
                elif 'timeout' in error_lower:
                    error_types['超时错误'] += 1
                elif any(keyword in error_lower for keyword in ['file', 'no such', 'cannot open']):
                    error_types['文件错误'] += 1
                else:
                    error_types['其他错误'] += 1
            
            result = f"错误分析报告 - '{log_file}':\n"
            result += "=" * 40 + "\n"
            result += f"总错误数: {len(errors)}\n\n"
            
            result += "错误类型分布:\n"
            result += "-" * 15 + "\n"
            for error_type, count in error_types.items():
                if count > 0:
                    percentage = (count / len(errors) * 100)
                    result += f"{error_type}: {count} ({percentage:.1f}%)\n"
            
            result += "\n典型错误示例 (前5个):\n"
            result += "-" * 20 + "\n"
            for error in errors[:5]:
                result += f"- {truncate_string(error, 100)}\n"
            
            if len(errors) > 5:
                result += f"\n... 还有 {len(errors) - 5} 个错误"
            
            # 建议
            result += "\n\n💡 修复建议:\n"
            if error_types['连接错误'] > 0:
                result += "- 检查网络连接和防火墙设置\n"
            if error_types['权限错误'] > 0:
                result += "- 检查文件和目录权限\n"
            if error_types['超时错误'] > 0:
                result += "- 增加超时时间或优化性能\n"
            if error_types['文件错误'] > 0:
                result += "- 验证文件路径和磁盘空间\n"
            
            return result
            
        except Exception as e:
            logger.error(f"错误分析失败: {e}")
            raise LogAnalysisError(f"无法分析错误日志 '{log_file}': {str(e)}")
    
    def _analyze_warnings(self, log_file: str) -> str:
        """分析警告日志"""
        try:
            if not self._validate_log_file(log_file):
                return f"无法访问日志文件 '{log_file}'。"
            
            content = self._read_log_file(log_file)
            
            config = LogAnalyzerConfig()
            warning_count = sum(1 for line in content.splitlines() if any(pattern in line.lower() for pattern in config.warning_patterns))
            
            warnings = [line for line in content.splitlines() if any(pattern in line.lower() for pattern in config.warning_patterns)]
            
            result = f"警告分析报告 - '{log_file}':\n"
            result += "=" * 40 + "\n"
            result += f"总警告数: {warning_count}\n\n"
            
            if warnings:
                result += "最近10个警告:\n"
                result += "-" * 15 + "\n"
                for warning in warnings[-10:]:
                    result += f"- {truncate_string(warning, 100)}\n"
            else:
                result += "✅ 未发现警告日志。"
            
            return result
            
        except Exception as e:
            logger.error(f"警告分析失败: {e}")
            raise LogAnalysisError(f"无法分析警告日志 '{log_file}': {str(e)}")
    
    def _generate_report(self, log_file: str) -> str:
        """生成日志报告"""
        try:
            if not self._validate_log_file(log_file):
                return f"无法访问日志文件 '{log_file}'。"
            
            content = self._read_log_file(log_file)
            stats = parse_log_level(content)
            errors = extract_error_patterns(content)
            
            total_lines = len(content.splitlines())
            total_errors = len(errors)
            
            result = f"日志分析报告 - '{log_file}'\n"
            result += "=" * 50 + "\n"
            result += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            result += f"文件大小: {self._get_file_size(log_file)}\n"
            result += f"总行数: {total_lines}\n\n"
            
            result += "1. 日志级别分布:\n"
            result += "-" * 20 + "\n"
            for level, count in stats.items():
                percentage = (count / total_lines * 100) if total_lines > 0 else 0
                result += f"{level}: {count} 行 ({percentage:.1f}%)\n"
            
            result += f"\n2. 错误统计: {total_errors} 个\n"
            if errors:
                result += "错误模式摘要:\n"
                error_summary = {}
                for error in errors:
                    key = error.split(':')[0].lower() if ':' in error else '其他'
                    error_summary[key] = error_summary.get(key, 0) + 1
                
                for error_type, count in sorted(error_summary.items(), key=lambda x: x[1], reverse=True)[:5]:
                    result += f"  - {error_type}: {count} 次\n"
            
            result += "\n3. 健康状态:\n"
            result += "-" * 15 + "\n"
            if total_errors == 0:
                result += "🟢 健康 - 未发现错误\n"
            elif total_errors < 10:
                result += "🟡 警告 - 发现少量错误\n"
            else:
                result += "🔴 问题 - 发现大量错误，需要关注\n"
            
            # 性能指标
            result += f"\n4. 性能指标:\n"
            result += f"   - ERROR率: {(total_errors / total_lines * 100):.2f}%\n"
            result += f"   - 最近1小时日志行数: {self._count_recent_logs(content, hours=1)}\n"
            
            return result
            
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            raise LogAnalysisError(f"无法生成日志报告 '{log_file}': {str(e)}")
    
    def _search_log(self, log_file: str, keyword: str) -> str:
        """搜索日志中的关键词"""
        try:
            if not self._validate_log_file(log_file):
                return f"无法访问日志文件 '{log_file}'。"
            
            content = self._read_log_file(log_file)
            lines = content.splitlines()
            
            matches = []
            for i, line in enumerate(lines, 1):
                if keyword.lower() in line.lower():
                    matches.append((i, line.strip()))
            
            if not matches:
                return f"在 '{log_file}' 中未找到包含 '{keyword}' 的日志。"
            
            result = f"搜索结果 - '{log_file}' (关键词: '{keyword}'):\n"
            result += "=" * 50 + "\n"
            result += f"找到 {len(matches)} 处匹配:\n\n"
            
            for line_num, line in matches[:20]:  # 显示前20个匹配
                result += f"第 {line_num} 行: {truncate_string(line, 120)}\n"
            
            if len(matches) > 20:
                result += f"\n... 还有 {len(matches) - 20} 处匹配"
            
            return result
            
        except Exception as e:
            logger.error(f"搜索日志失败: {e}")
            raise LogAnalysisError(f"无法搜索日志 '{log_file}': {str(e)}")
    
    def _search_all_logs(self, keyword: str) -> str:
        """搜索所有常见日志文件"""
        common_logs = [
            '/var/log/syslog', '/var/log/messages', '/var/log/auth.log',
            '/var/log/nginx/error.log', '/var/log/nginx/access.log',
            '/var/log/apache2/error.log', '/var/log/mysql/error.log'
        ]
        
        results = []
        for log_file in common_logs:
            if Path(log_file).exists():
                try:
                    content = self._read_log_file(log_file)
                    lines = content.splitlines()
                    matches = sum(1 for line in lines if keyword.lower() in line.lower())
                    if matches > 0:
                        results.append(f"{log_file}: {matches} 处匹配")
                except Exception:
                    continue
        
        if not results:
            return f"未在常见日志文件中找到包含 '{keyword}' 的记录。"
        
        result = f"跨日志搜索结果 (关键词: '{keyword}'):\n"
        result += "=" * 40 + "\n"
        for res in results:
            result += f"- {res}\n"
        
        return result
    
    def _get_log_statistics(self, log_file: str) -> str:
        """获取日志统计信息"""
        try:
            if not self._validate_log_file(log_file):
                return f"无法访问日志文件 '{log_file}'。"
            
            content = self._read_log_file(log_file)
            stats = parse_log_level(content)
            
            total_lines = len(content.splitlines())
            
            result = f"日志统计 - '{log_file}':\n"
            result += "=" * 30 + "\n"
            result += f"总行数: {total_lines}\n\n"
            
            result += "按级别统计:\n"
            result += "-" * 15 + "\n"
            for level, count in stats.items():
                percentage = (count / total_lines * 100) if total_lines > 0 else 0
                result += f"{level}: {count} ({percentage:.1f}%)\n"
            
            # 时间分布（如果日志有时间戳）
            time_pattern = r'(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})'
            timestamps = re.findall(time_pattern, content)
            if timestamps:
                recent_hour = sum(1 for ts in timestamps if datetime.now() - datetime.fromisoformat(ts.replace(' ', 'T')) < timedelta(hours=1))
                result += f"\n最近1小时日志: {recent_hour} 行 ({recent_hour / total_lines * 100:.1f}%)"
            
            return result
            
        except Exception as e:
            logger.error(f"获取日志统计失败: {e}")
            raise LogAnalysisError(f"无法获取日志统计 '{log_file}': {str(e)}")
    
    def _validate_log_file(self, log_file: str) -> bool:
        """验证日志文件"""
        path = Path(log_file)
        if not path.exists():
            return False
        
        if not path.is_file():
            return False
        
        config = LogAnalyzerConfig()
        max_size_bytes = self._parse_size(config.max_file_size)
        
        if path.stat().st_size > max_size_bytes:
            logger.warning(f"日志文件 '{log_file}' 太大 ({path.stat().st_size} > {max_size_bytes})")
            return False
        
        return True
    
    def _read_log_file(self, log_file: str, max_lines: int = 10000) -> str:
        """读取日志文件（限制行数）"""
        try:
            path = Path(log_file)
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[-max_lines:]  # 读取最后max_lines行
                return ''.join(lines)
        except Exception as e:
            logger.error(f"读取日志文件失败: {e}")
            raise FileOperationError(f"无法读取日志文件 '{log_file}': {str(e)}")
    
    def _get_file_size(self, log_file: str) -> str:
        """获取文件大小"""
        try:
            path = Path(log_file)
            return f"{path.stat().st_size / 1024 / 1024:.2f} MB"
        except:
            return "未知"
    
    def _parse_size(self, size_str: str) -> int:
        """解析大小字符串"""
        size_str = size_str.upper().strip()
        if size_str.endswith('KB'):
            return int(float(size_str[:-2]) * 1024)
        elif size_str.endswith('MB'):
            return int(float(size_str[:-2]) * 1024 * 1024)
        elif size_str.endswith('GB'):
            return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
        else:
            return int(size_str)
    
    def _count_recent_logs(self, content: str, hours: int = 1) -> int:
        """统计最近hours小时的日志行数"""
        time_pattern = r'(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})'
        timestamps = re.findall(time_pattern, content)
        
        now = datetime.now()
        recent_count = 0
        
        for ts_str in timestamps:
            try:
                ts = datetime.fromisoformat(ts_str.replace(' ', 'T'))
                if now - ts < timedelta(hours=hours):
                    recent_count += 1
            except ValueError:
                continue
        
        return recent_count


if __name__ == "__main__":
    # 测试日志分析工具
    try:
        tool = LogAnalyzerTool()
        print("测试日志分析工具:")
        # 由于没有实际日志文件，这里测试通用分析
        print(tool._run("分析日志文件中的错误"))
    except Exception as e:
        print(f"测试失败: {e}")
