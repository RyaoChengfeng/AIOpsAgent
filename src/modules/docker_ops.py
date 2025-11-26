"""
Docker操作模块
提供Docker相关的工具函数和LangChain工具
"""

import docker
from typing import Dict, List, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from config.settings import get_config
from src.utils.logger import get_logger
from src.utils.helpers import run_command, format_duration
from src.utils.exceptions import DockerOperationError, CommandExecutionError

logger = get_logger(__name__)


class DockerConfig(BaseModel):
    """Docker配置模型"""
    socket_path: str = Field(default_factory=lambda: get_config('docker.socket_path', 'tcp://localhost:2375'))
    timeout: int = Field(default_factory=lambda: get_config('docker.timeout', 30))
    api_version: str = Field(default_factory=lambda: get_config('docker.api_version', 'auto'))


class DockerOpsTool(BaseTool):
    """Docker操作LangChain工具"""

    name: str = "docker_operations"
    description: str = (
        "用于执行Docker容器和镜像操作的工具。支持启动、停止、重启容器，"
        "查询容器状态，拉取和删除镜像，查看容器日志等操作。"
        "输入应为具体的Docker命令描述，如'启动名为web-app的容器'或'查看所有容器状态'"
    )
    args_schema: Optional[BaseModel] = None

    def _run(self, command: str) -> str:
        """
        执行Docker操作
        
        Args:
            command: Docker命令描述
            
        Returns:
            操作结果
        """
        try:
            client = self._get_docker_client()

            # 解析命令意图
            action, target = self._parse_command(command)

            if action == "list_containers":
                return self._list_containers(client)
            elif action == "container_status":
                return self._get_container_status(client, target)
            elif action == "start_container":
                return self._start_container(client, target)
            elif action == "stop_container":
                return self._stop_container(client, target)
            elif action == "restart_container":
                return self._restart_container(client, target)
            elif action == "container_logs":
                return self._get_container_logs(client, target)
            elif action == "pull_image":
                return self._pull_image(client, target)
            elif action == "remove_image":
                return self._remove_image(client, target)
            elif action == "list_images":
                return self._list_images(client)
            else:
                return f"不支持的Docker操作: {command}。支持的操作包括: 列出容器、容器状态、启动/停止/重启容器、容器日志、拉取/删除镜像、列出镜像。"

        except Exception as e:
            logger.error(f"Docker操作失败: {e}")
            raise DockerOperationError(f"Docker操作执行失败: {str(e)}")

    def _get_docker_client(self) -> docker.DockerClient:
        """获取Docker客户端"""
        config = DockerConfig()
        try:
            client = docker.DockerClient(
                base_url=config.socket_path,
                version=config.api_version,
                timeout=config.timeout
            )
            # 测试连接
            client.ping()
            return client
        except Exception as e:
            # 尝试使用命令行fallback
            logger.warning(f"Docker API连接失败，使用命令行模式: {e}")
            return None

    def _parse_command(self, command: str) -> tuple:
        """解析用户命令"""
        command_lower = command.lower()

        if "列出容器" in command_lower or "list containers" in command_lower:
            return "list_containers", None
        elif "状态" in command_lower or "status" in command_lower:
            target = command.split()[-1] if command.split() else None
            return "container_status", target
        elif "启动" in command_lower or "start" in command_lower:
            target = command.split()[-1] if command.split() else None
            return "start_container", target
        elif "停止" in command_lower or "stop" in command_lower:
            target = command.split()[-1] if command.split() else None
            return "stop_container", target
        elif "重启" in command_lower or "restart" in command_lower:
            target = command.split()[-1] if command.split() else None
            return "restart_container", target
        elif "日志" in command_lower or "logs" in command_lower:
            target = command.split()[-1] if command.split() else None
            return "container_logs", target
        elif "拉取" in command_lower or "pull" in command_lower:
            target = command.replace("拉取", "").replace("pull", "").strip()
            return "pull_image", target
        elif "删除镜像" in command_lower or "remove image" in command_lower:
            target = command.split()[-1] if command.split() else None
            return "remove_image", target
        elif "列出镜像" in command_lower or "list images" in command_lower:
            return "list_images", None
        else:
            return "unknown", command

    def _list_containers(self, client) -> str:
        """列出所有容器"""
        if client:
            try:
                containers = client.containers.list(all=True)
                if not containers:
                    return "当前没有运行或停止的容器。"

                result = "Docker容器列表:\\n"
                result += "容器ID\\t\\t名称\\t\\t状态\\t\\t镜像\\n"
                result += "-" * 50 + "\\n"

                for container in containers:
                    status = container.status
                    name = container.name or "无名称"
                    image = container.image.tags[0] if container.image.tags else "未知镜像"
                    result += f"{container.short_id}\\t{name}\\t\\t{status}\\t\\t{image}\\n"

                return result
            except Exception as e:
                logger.error(f"列出容器失败: {e}")
                raise DockerOperationError("无法列出容器")
        else:
            # Fallback到命令行
            return self._run_docker_command(
                "docker ps -a --format 'table {{.ID}}\\t{{.Names}}\\t{{.Status}}\\t{{.Image}}'")

    def _get_container_status(self, client, container_name: str) -> str:
        """获取容器状态"""
        if not container_name:
            return "请指定容器名称。"

        if client:
            try:
                container = client.containers.get(container_name)
                info = container.attrs
                status = info['State']['Status']
                uptime = format_duration(int(info['State']['StartedAt']) / 1000000000) if 'StartedAt' in info[
                    'State'] else "未知"

                result = f"容器 '{container_name}' 状态信息:\\n"
                result += f"状态: {status}\\n"
                result += f"运行时间: {uptime}\\n"
                result += f"镜像: {info['Config']['Image']}\\n"
                result += f"端口映射: {info['HostConfig']['PortBindings']}\\n"
                result += f"CPU使用率: {info['State']['CpuUsage']['UsageInKernelmode'] / 10 ** 9:.1f}%\\n"
                result += f"内存使用: {info['State']['MemoryStats']['usage'] / 1024 ** 2:.1f}MB"

                return result
            except docker.errors.NotFound:
                return f"容器 '{container_name}' 不存在。"
            except Exception as e:
                logger.error(f"获取容器状态失败: {e}")
                raise DockerOperationError(f"无法获取容器 '{container_name}' 状态")
        else:
            return self._run_docker_command(f"docker inspect {container_name} --format='{{json .State}}'")

    def _start_container(self, client, container_name: str) -> str:
        """启动容器"""
        if not container_name:
            return "请指定要启动的容器名称。"

        if client:
            try:
                container = client.containers.get(container_name)
                if container.status == 'running':
                    return f"容器 '{container_name}' 已经在运行。"

                container.start()
                return f"✅ 容器 '{container_name}' 启动成功！\\n容器ID: {container.id[:12]}\\n状态: Running"
            except docker.errors.NotFound:
                return f"容器 '{container_name}' 不存在。请先创建容器。"
            except Exception as e:
                logger.error(f"启动容器失败: {e}")
                raise DockerOperationError(f"无法启动容器 '{container_name}': {str(e)}")
        else:
            result = self._run_docker_command(f"docker start {container_name}")
            if result[0] == 0:
                return f"✅ 容器 '{container_name}' 启动成功！"
            else:
                return f"❌ 启动容器 '{container_name}' 失败: {result[2]}"

    def _stop_container(self, client, container_name: str) -> str:
        """停止容器"""
        if not container_name:
            return "请指定要停止的容器名称。"

        if client:
            try:
                container = client.containers.get(container_name)
                if container.status != 'running':
                    return f"容器 '{container_name}' 没有在运行。"

                container.stop()
                return f"✅ 容器 '{container_name}' 已停止。\\n容器ID: {container.id[:12]}"
            except docker.errors.NotFound:
                return f"容器 '{container_name}' 不存在。"
            except Exception as e:
                logger.error(f"停止容器失败: {e}")
                raise DockerOperationError(f"无法停止容器 '{container_name}': {str(e)}")
        else:
            result = self._run_docker_command(f"docker stop {container_name}")
            if result[0] == 0:
                return f"✅ 容器 '{container_name}' 已停止。"
            else:
                return f"❌ 停止容器 '{container_name}' 失败: {result[2]}"

    def _restart_container(self, client, container_name: str) -> str:
        """重启容器"""
        if not container_name:
            return "请指定要重启的容器名称。"

        if client:
            try:
                container = client.containers.get(container_name)
                container.restart()
                return f"✅ 容器 '{container_name}' 已重启。\\n容器ID: {container.id[:12]}\\n状态: Running"
            except docker.errors.NotFound:
                return f"容器 '{container_name}' 不存在。"
            except Exception as e:
                logger.error(f"重启容器失败: {e}")
                raise DockerOperationError(f"无法重启容器 '{container_name}': {str(e)}")
        else:
            result = self._run_docker_command(f"docker restart {container_name}")
            if result[0] == 0:
                return f"✅ 容器 '{container_name}' 已重启。"
            else:
                return f"❌ 重启容器 '{container_name}' 失败: {result[2]}"

    def _get_container_logs(self, client, container_name: str, lines: int = 50) -> str:
        """获取容器日志"""
        if not container_name:
            return "请指定容器名称。"

        if client:
            try:
                container = client.containers.get(container_name)
                logs = container.logs(tail=lines, stream=False).decode('utf-8')
                if not logs.strip():
                    return f"容器 '{container_name}' 没有日志输出。"

                # 截断长日志
                if len(logs) > 2000:
                    logs = logs[-2000:] + "\\n... (日志已截断)"

                return f"容器 '{container_name}' 最新 {lines} 行日志:\\n\\n{logs}"
            except docker.errors.NotFound:
                return f"容器 '{container_name}' 不存在。"
            except Exception as e:
                logger.error(f"获取容器日志失败: {e}")
                raise DockerOperationError(f"无法获取容器 '{container_name}' 日志")
        else:
            result = self._run_docker_command(f"docker logs --tail {lines} {container_name}")
            if result[0] == 0:
                return f"容器 '{container_name}' 最新 {lines} 行日志:\\n\\n{result[1]}"
            else:
                return f"❌ 获取容器 '{container_name}' 日志失败: {result[2]}"

    def _pull_image(self, client, image_name: str) -> str:
        """拉取Docker镜像"""
        if not image_name:
            return "请指定要拉取的镜像名称（如 nginx:latest）。"

        if client:
            try:
                result = client.images.pull(image_name)
                return f"✅ 镜像 '{image_name}' 拉取成功！\\n镜像ID: {result.id}"
            except Exception as e:
                logger.error(f"拉取镜像失败: {e}")
                raise DockerOperationError(f"无法拉取镜像 '{image_name}': {str(e)}")
        else:
            result = self._run_docker_command(f"docker pull {image_name}")
            if result[0] == 0:
                return f"✅ 镜像 '{image_name}' 拉取成功！"
            else:
                return f"❌ 拉取镜像 '{image_name}' 失败: {result[2]}"

    def _remove_image(self, client, image_name: str) -> str:
        """删除Docker镜像"""
        if not image_name:
            return "请指定要删除的镜像名称。"

        if client:
            try:
                client.images.remove(image_name, force=True)
                return f"✅ 镜像 '{image_name}' 已删除。"
            except Exception as e:
                logger.error(f"删除镜像失败: {e}")
                raise DockerOperationError(f"无法删除镜像 '{image_name}': {str(e)}")
        else:
            result = self._run_docker_command(f"docker rmi -f {image_name}")
            if result[0] == 0:
                return f"✅ 镜像 '{image_name}' 已删除。"
            else:
                return f"❌ 删除镜像 '{image_name}' 失败: {result[2]}"

    def _list_images(self, client) -> str:
        """列出所有镜像"""
        if client:
            try:
                images = client.images.list(all=True)
                if not images:
                    return "当前没有Docker镜像。"

                result = "Docker镜像列表:\\n"
                result += "镜像ID\\t\\t仓库\\t\\t标签\\t\\t大小\\n"
                result += "-" * 40 + "\\n"

                for image in images:
                    repo_tags = image.tags if image.tags else ["<none>:<none>"]
                    size = format_bytes(image.attrs['Size'])
                    result += f"{image.short_id}\\t{repo_tags[0].split(':')[0]}\\t{repo_tags[0].split(':')[1]}\\t{size}\\n"

                return result
            except Exception as e:
                logger.error(f"列出镜像失败: {e}")
                raise DockerOperationError("无法列出镜像")
        else:
            return self._run_docker_command(
                "docker images --format 'table {{.ID}}\\t{{.Repository}}\\t{{.Tag}}\\t{{.Size}}'")

    def _run_docker_command(self, cmd: str) -> tuple:
        """执行Docker命令行命令（fallback）"""
        try:
            return run_command(f"docker {cmd}", timeout=60)
        except Exception as e:
            raise CommandExecutionError(f"Docker命令执行失败: {str(e)}", f"docker {cmd}")


if __name__ == "__main__":
    # 测试Docker工具
    try:
        tool = DockerOpsTool()
        print("测试Docker工具:")
        print(tool._list_containers(None))
    except Exception as e:
        print(f"测试失败: {e}")
