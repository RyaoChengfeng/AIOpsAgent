"""
日志分析模块
提供日志文件解析和分析的工具函数和LangChain工具
"""

import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from config.settings import get_config
from src.utils.logger import get_logger
from src.utils.helpers import (
    parse_log_level, extract_error_patterns, safe_json_loads,
    truncate_string, run_command, format_duration
)
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
    # 新增：支持的时间格式
    time_formats: List[str] = Field(default_factory=lambda: [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%b %d %H:%M:%S",
        "%d/%b/%Y:%H:%M:%S"
    ])


class LogAnalyzerTool(BaseTool):
    """日志分析LangChain工具"""

    name: str = "log_analyzer"
    description: str = (
        "用于分析日志文件的工具。支持解析服务日志，搜索错误或异常，"
        "自动分类日志内容并生成报告，统计日志级别等操作。"
        "支持按时间范围分析（如'分析最近1小时的错误'）和JSON格式日志解析。"
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

            # 新增：时间范围分析支持
            time_range = self._extract_time_range(analysis_request)

            if "分析" in request_lower or "analyze" in request_lower:
                log_file = self._extract_log_file(analysis_request)
                if log_file:
                    # 传递时间范围参数到分析方法
                    if "错误" in request_lower or "error" in request_lower:
                        return self._analyze_errors(log_file, time_range)
                    elif "警告" in request_lower or "warning" in request_lower:
                        return self._analyze_warnings(log_file, time_range)
                    elif "报告" in request_lower or "report" in request_lower:
                        return self._generate_report(log_file, time_range)
                    else:
                        return self._analyze_log_file(log_file, time_range)
                else:
                    return "请指定日志文件路径。"
            elif "搜索" in request_lower or "search" in request_lower:
                keyword = self._extract_keyword(analysis_request)
                log_file = self._extract_log_file(analysis_request)
                if log_file and keyword:
                    return self._search_log(log_file, keyword, time_range)
                elif keyword:
                    return self._search_all_logs(keyword, time_range)
                else:
                    return "请指定搜索关键词和日志文件。"
            elif "统计" in request_lower or "statistics" in request_lower:
                log_file = self._extract_log_file(analysis_request)
                if log_file:
                    return self._get_log_statistics(log_file, time_range)
                else:
                    return "请指定日志文件。"
            else:
                return (
                    "支持的日志分析操作:\\n"
                    "- 分析日志文件 (指定路径，可加时间范围如'最近1小时')\\n"
                    "- 搜索日志中的特定关键词 (指定文件和关键词)\\n"
                    "- 生成日志报告 (指定文件)\\n"
                    "- 统计日志级别分布 (指定文件)\\n"
                    "- 分析错误/警告日志 (指定文件)\\n"
                    "示例: '分析/var/log/nginx/error.log中最近1小时的错误'"
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

    def _extract_time_range(self, request: str) -> Optional[Tuple[datetime, datetime]]:
        """
        从请求中提取时间范围
        支持格式: "最近1小时", "过去2天", "今天", "昨天"
        返回: (start_time, end_time) 或 None
        """
        request_lower = request.lower()
        end_time = datetime.now()
        start_time = None

        if "最近" in request_lower or "过去" in request_lower:
            # 提取数字和单位
            match = re.search(r'(\d+)\s*(小时|天|分钟|秒)', request_lower)
            if match:
                num = int(match.group(1))
                unit = match.group(2)

                if unit == "小时":
                    start_time = end_time - timedelta(hours=num)
                elif unit == "天":
                    start_time = end_time - timedelta(days=num)
                elif unit == "分钟":
                    start_time = end_time - timedelta(minutes=num)
                elif unit == "秒":
                    start_time = end_time - timedelta(seconds=num)

        elif "今天" in request_lower:
            start_time = datetime(end_time.year, end_time.month, end_time.day)

        elif "昨天" in request_lower:
            yesterday = end_time - timedelta(days=1)
            start_time = datetime(yesterday.year, yesterday.month, yesterday.day)
            end_time = datetime(end_time.year, end_time.month, end_time.day)

        return (start_time, end_time) if start_time else None

    def _parse_log_time(self, line: str) -> Optional[datetime]:
        """解析日志行中的时间戳"""
        config = LogAnalyzerConfig()
        for fmt in config.time_formats:
            try:
                # 尝试从行首提取时间
                time_str = line[:20].strip()  # 取行首20字符尝试解析
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue

        # 尝试JSON日志格式
        json_data = safe_json_loads(line)
        if json_data and isinstance(json_data, dict):
            for key in ['time', 'timestamp', 'log_time']:
                if key in json_data:
                    try:
                        return datetime.fromisoformat(str(json_data[key]).replace('Z', '+00:00'))
                    except ValueError:
                        continue

        return None

    def _filter_logs_by_time(self, content: str, time_range: Tuple[datetime, datetime]) -> str:
        """根据时间范围过滤日志内容"""
        start_time, end_time = time_range
        filtered_lines = []

        for line in content.splitlines():
            log_time = self._parse_log_time(line)
            if log_time and start_time <= log_time <= end_time:
                filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def _analyze_log_file(self, log_file: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """分析日志文件（支持时间范围过滤）"""
        try:
            if not self._validate_log_file(log_file):
                return f"无法访问日志文件 '{log_file}' 或文件太大。"

            content = self._read_log_file(log_file)
            if not content:
                return f"日志文件 '{log_file}' 为空或无法读取。"

            # 应用时间范围过滤
            filtered_content = content
            time_filter_info = ""
            if time_range:
                filtered_content = self._filter_logs_by_time(content, time_range)
                time_filter_info = f"（时间范围: {time_range[0]} 至 {time_range[1]}）"
                if not filtered_content:
                    return f"在指定时间范围内未找到日志内容 {time_filter_info}"

            # 基本统计
            stats = parse_log_level(filtered_content)
            errors = extract_error_patterns(filtered_content)

            config = LogAnalyzerConfig()
            max_size_bytes = self._parse_size(config.max_file_size)

            result = f"日志文件 '{log_file}' 分析报告 {time_filter_info}:\n"
            result += "=" * 50 + "\n"
            result += f"文件大小: {self._get_file_size(log_file)}\n"
            result += f"总行数: {len(content.splitlines())}\n"
            result += f"过滤后行数: {len(filtered_content.splitlines())}\n\n"

            result += "日志级别统计:\n"
            result += "-" * 20 + "\n"
            total = sum(stats.values())
            for level, count in stats.items():
                percentage = (count / total * 100) if total > 0 else 0
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
            lines = filtered_content.splitlines()
            recent_logs = '\n'.join(lines[-10:]) if lines else "无日志内容"
            result += f"\n最近10行日志:\n"
            result += "-" * 20 + "\n"
            result += truncate_string(recent_logs, 1000)

            return result

        except Exception as e:
            logger.error(f"分析日志文件失败: {e}")
            raise LogAnalysisError(f"无法分析日志文件 '{log_file}': {str(e)}")

    def _analyze_errors(self, log_file: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """分析错误日志（支持时间范围过滤和频率统计）"""
        try:
            if not self._validate_log_file(log_file):
                return f"无法访问日志文件 '{log_file}'。"

            content = self._read_log_file(log_file)

            # 应用时间范围过滤
            filtered_content = content
            time_filter_info = ""
            if time_range:
                filtered_content = self._filter_logs_by_time(content, time_range)
                time_filter_info = f"（时间范围: {time_range[0]} 至 {time_range[1]}）"
                if not filtered_content:
                    return f"在指定时间范围内未找到日志内容 {time_filter_info}"

            errors = extract_error_patterns(filtered_content)

            if not errors:
                return f"✅ 在日志文件 '{log_file}' {time_filter_info}中未发现错误。"

            # 分类错误
            error_types = {
                '连接错误': 0,
                '权限错误': 0,
                '超时错误': 0,
                '文件错误': 0,
                '其他错误': 0
            }

            # 错误时间分布统计
            error_timestamps = []
            for error in errors:
                error_lower = error.lower()
                # 分类统计
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

                # 提取错误时间戳
                log_line = self._find_error_line(content, error)
                if log_line:
                    log_time = self._parse_log_time(log_line)
                    if log_time:
                        error_timestamps.append(log_time)

            # 生成错误频率报告
            frequency_report = ""
            if error_timestamps:
                error_timestamps.sort()
                # 按小时统计
                hourly_counts = {}
                for dt in error_timestamps:
                    hour_key = dt.strftime("%Y-%m-%d %H:00")
                    hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1

                frequency_report = "\n错误频率（按小时）:\n"
                frequency_report += "-" * 25 + "\n"
                # 取最近6小时
                recent_hours = sorted(hourly_counts.keys())[-6:]
                for hour in recent_hours:
                    frequency_report += f"{hour}: {hourly_counts[hour]} 次\n"

            # 标记关键错误
            critical_errors = [e for e in errors if any(kw in e.lower() for kw in ['fatal', 'critical', 'panic'])]
            critical_note = f"\n⚠️  发现 {len(critical_errors)} 个致命错误，可能影响服务可用性！" if critical_errors else ""

            result = f"错误分析报告 - '{log_file}' {time_filter_info}:\n"
            result += "=" * 40 + "\n"
            result += f"总错误数: {len(errors)}\n{critical_note}\n\n"

            result += "错误类型分布:\n"
            result += "-" * 15 + "\n"
            total_errors = len(errors)
            for error_type, count in error_types.items():
                if count > 0:
                    percentage = (count / total_errors * 100)
                    result += f"{error_type}: {count} ({percentage:.1f}%)\n"

            result += frequency_report

            result += "\n典型错误示例 (前5个):\n"
            result += "-" * 20 + "\n"
            for error in errors[:5]:
                result += f"- {truncate_string(error, 100)}\n"

            if len(errors) > 5:
                result += f"\n... 还有 {len(errors) - 5} 个错误"

            return result

        except Exception as e:
            logger.error(f"分析错误日志失败: {e}")
            raise LogAnalysisError(f"无法分析 '{log_file}' 中的错误: {str(e)}")

    def _analyze_warnings(self, log_file: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """分析警告日志（新增方法，支持时间范围）"""
        try:
            if not self._validate_log_file(log_file):
                return f"无法访问日志文件 '{log_file}'。"

            content = self._read_log_file(log_file)

            # 应用时间范围过滤
            filtered_content = content
            time_filter_info = ""
            if time_range:
                filtered_content = self._filter_logs_by_time(content, time_range)
                time_filter_info = f"（时间范围: {time_range[0]} 至 {time_range[1]}）"
                if not filtered_content:
                    return f"在指定时间范围内未找到日志内容 {time_filter_info}"

            # 提取警告信息
            config = LogAnalyzerConfig()
            warning_pattern = re.compile('|'.join(config.warning_patterns), re.IGNORECASE)
            warnings = [line.strip() for line in filtered_content.splitlines() if warning_pattern.search(line)]

            if not warnings:
                return f"✅ 在日志文件 '{log_file}' {time_filter_info}中未发现警告。"

            result = f"警告分析报告 - '{log_file}' {time_filter_info}:\n"
            result += "=" * 40 + "\n"
            result += f"总警告数: {len(warnings)}\n\n"

            # 分类警告
            deprecation_warnings = [w for w in warnings if 'deprecated' in w.lower()]
            config_warnings = [w for w in warnings if 'config' in w.lower() or 'configuration' in w.lower()]
            other_warnings = [w for w in warnings if w not in deprecation_warnings and w not in config_warnings]

            result += "警告类型分布:\n"
            result += "-" * 15 + "\n"
            result += f"废弃警告: {len(deprecation_warnings)} ({len(deprecation_warnings)/len(warnings)*100:.1f}%)\n"
            result += f"配置警告: {len(config_warnings)} ({len(config_warnings)/len(warnings)*100:.1f}%)\n"
            result += f"其他警告: {len(other_warnings)} ({len(other_warnings)/len(warnings)*100:.1f}%)\n"

            result += "\n典型警告示例 (前5个):\n"
            result += "-" * 20 + "\n"
            for warning in warnings[:5]:
                result += f"- {truncate_string(warning, 100)}\n"

            if len(warnings) > 5:
                result += f"\n... 还有 {len(warnings) - 5} 个警告"

            return result

        except Exception as e:
            logger.error(f"分析警告日志失败: {e}")
            raise LogAnalysisError(f"无法分析 '{log_file}' 中的警告: {str(e)}")

    def _generate_report(self, log_file: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """生成日志报告（增强版）"""
        try:
            if not self._validate_log_file(log_file):
                return f"无法访问日志文件 '{log_file}'。"

            # 整合各类分析结果
            analysis = self._analyze_log_file(log_file, time_range)
            errors = self._analyze_errors(log_file, time_range)
            warnings = self._analyze_warnings(log_file, time_range)

            # 提取关键指标
            error_count = re.search(r'总错误数: (\d+)', errors).group(1) if re.search(r'总错误数: (\d+)', errors) else '0'
            warning_count = re.search(r'总警告数: (\d+)', warnings).group(1) if re.search(r'总警告数: (\d+)', warnings) else '0'

            # 生成摘要
            summary = f"日志综合报告 - '{log_file}'\n"
            summary += "=" * 60 + "\n"
            summary += f"关键指标: 错误数={error_count}, 警告数={warning_count}\n\n"
            summary += "详细分析:\n"
            summary += "-" * 20 + "\n"
            summary += "1. 基本分析:\n"
            summary += analysis.split('最近10行日志:')[0]  # 截断过长内容
            summary += "\n2. 错误分析:\n"
            summary += errors.split('典型错误示例')[0]
            summary += "\n3. 警告分析:\n"
            summary += warnings.split('典型警告示例')[0]

            return truncate_string(summary, 3000)  # 限制报告长度

        except Exception as e:
            logger.error(f"生成日志报告失败: {e}")
            raise LogAnalysisError(f"无法为 '{log_file}' 生成报告: {str(e)}")

    def _search_log(self, log_file: str, keyword: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """搜索日志（支持时间范围）"""
        try:
            if not self._validate_log_file(log_file):
                return f"无法访问日志文件 '{log_file}'。"

            content = self._read_log_file(log_file)

            # 应用时间范围过滤
            filtered_content = content
            time_filter_info = ""
            if time_range:
                filtered_content = self._filter_logs_by_time(content, time_range)
                time_filter_info = f"（时间范围: {time_range[0]} 至 {time_range[1]}）"

            # 搜索关键词
            lines = filtered_content.splitlines()
            matches = [line for line in lines if keyword.lower() in line.lower()]

            if not matches:
                return f"在日志文件 '{log_file}' 中未找到包含 '{keyword}' 的内容 {time_filter_info}。"

            result = f"在 '{log_file}' 中找到 {len(matches)} 条包含 '{keyword}' 的记录 {time_filter_info}:\n"
            result += "-" * 50 + "\n"

            # 显示前10条匹配结果
            for line in matches[:10]:
                # 提取时间戳（如果有）
                log_time = self._parse_log_time(line)
                time_str = log_time.strftime("%Y-%m-%d %H:%M:%S") if log_time else "未知时间"
                result += f"[{time_str}] {truncate_string(line, 120)}\n"

            if len(matches) > 10:
                result += f"\n... 还有 {len(matches) - 10} 条匹配记录"

            return result

        except Exception as e:
            logger.error(f"搜索日志失败: {e}")
            raise LogAnalysisError(f"无法在 '{log_file}' 中搜索 '{keyword}': {str(e)}")

    def _search_all_logs(self, keyword: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """搜索所有常见日志（支持时间范围）"""
        common_logs = [
            '/var/log/nginx/error.log', '/var/log/nginx/access.log',
            '/var/log/apache2/error.log', '/var/log/apache2/access.log',
            '/var/log/mysql/error.log', '/var/log/syslog',
            '/var/log/auth.log', '/var/log/kern.log'
        ]

        result = f"搜索所有日志中包含 '{keyword}' 的记录:\n"
        result += "-" * 50 + "\n"

        for log_file in common_logs:
            try:
                if not self._validate_log_file(log_file):
                    continue

                content = self._read_log_file(log_file)

                # 应用时间范围过滤
                filtered_content = content
                if time_range:
                    filtered_content = self._filter_logs_by_time(content, time_range)

                matches = [line for line in filtered_content.splitlines() if keyword.lower() in line.lower()]
                if matches:
                    result += f"\n在 {log_file} 中找到 {len(matches)} 条记录:\n"
                    result += "-" * 30 + "\n"
                    for line in matches[:3]:  # 每个文件显示前3条
                        result += f"- {truncate_string(line, 100)}\n"
            except Exception as e:
                logger.warning(f"搜索日志 {log_file} 失败: {e}")
                result += f"\n⚠️  搜索 {log_file} 时出错: {str(e)}\n"

        return result

    def _get_log_statistics(self, log_file: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """获取日志统计信息（增强版）"""
        try:
            if not self._validate_log_file(log_file):
                return f"无法访问日志文件 '{log_file}'。"

            content = self._read_log_file(log_file)

            # 应用时间范围过滤
            filtered_content = content
            time_filter_info = ""
            if time_range:
                filtered_content = self._filter_logs_by_time(content, time_range)
                time_filter_info = f"（时间范围: {time_range[0]} 至 {time_range[1]}）"
                if not filtered_content:
                    return f"在指定时间范围内未找到日志内容 {time_filter_info}"

            # 基本统计
            lines = filtered_content.splitlines()
            level_stats = parse_log_level(filtered_content)
            error_count = len(extract_error_patterns(filtered_content))

            # 时间分布统计
            time_buckets = {}
            for line in lines:
                log_time = self._parse_log_time(line)
                if log_time:
                    # 按小时分组
                    bucket_key = log_time.strftime("%Y-%m-%d %H:00")
                    time_buckets[bucket_key] = time_buckets.get(bucket_key, 0) + 1

            result = f"日志统计信息 - '{log_file}' {time_filter_info}:\n"
            result += "=" * 50 + "\n"
            result += f"总行数: {len(lines)}\n"
            result += f"错误总数: {error_count}\n"
            result += f"日志覆盖时间段: {min(time_buckets.keys()) if time_buckets else '未知'} 至 {max(time_buckets.keys()) if time_buckets else '未知'}\n\n"

            result += "日志级别分布:\n"
            result += "-" * 20 + "\n"
            total = sum(level_stats.values())
            for level, count in level_stats.items():
                percentage = (count / total * 100) if total > 0 else 0
                result += f"{level}: {count} ({percentage:.1f}%)\n"

            if time_buckets:
                result += "\n日志量时间分布（按小时）:\n"
                result += "-" * 30 + "\n"
                # 取最近8小时
                sorted_buckets = sorted(time_buckets.items())[-8:]
                for bucket, count in sorted_buckets:
                    result += f"{bucket}: {count} 行\n"

            return result

        except Exception as e:
            logger.error(f"获取日志统计失败: {e}")
            raise LogAnalysisError(f"无法获取 '{log_file}' 的统计信息: {str(e)}")

    # 辅助方法
    def _validate_log_file(self, log_file: str) -> bool:
        """验证日志文件是否可访问且大小合法"""
        try:
            path = Path(log_file)
            if not path.exists() or not path.is_file():
                return False

            # 检查文件大小
            config = LogAnalyzerConfig()
            max_size = self._parse_size(config.max_file_size)
            if path.stat().st_size > max_size:
                return False

            return True
        except Exception:
            return False

    def _read_log_file(self, log_file: str) -> str:
        """读取日志文件内容"""
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"读取日志文件失败: {e}")
            raise FileOperationError(f"无法读取日志文件 '{log_file}': {str(e)}")

    def _parse_size(self, size_str: str) -> int:
        """解析大小字符串为字节数（复用logger中的实现思路）"""
        size_str = size_str.upper().strip()

        if size_str.endswith('KB'):
            return int(float(size_str[:-2]) * 1024)
        elif size_str.endswith('MB'):
            return int(float(size_str[:-2]) * 1024 * 1024)
        elif size_str.endswith('GB'):
            return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
        else:
            return int(size_str)

    def _get_file_size(self, log_file: str) -> str:
        """获取文件大小的人类可读格式"""
        try:
            size = Path(log_file).stat().st_size
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f}{unit}"
                size /= 1024.0
            return f"{size:.1f}TB"
        except Exception:
            return "未知"

    def _find_error_line(self, content: str, error_pattern: str) -> Optional[str]:
        """查找包含错误模式的完整日志行"""
        for line in content.splitlines():
            if error_pattern in line:
                return line
        return None
