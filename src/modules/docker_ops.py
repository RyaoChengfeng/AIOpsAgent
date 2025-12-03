"""
Docker operations module
Provides Docker-related utility functions and LangChain tools.
"""

import docker
from typing import Dict, List, Any, Optional

from langchain.chains import llm
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from config.settings import get_config
from src.utils.logger import get_logger
from src.utils.helpers import run_command, format_duration, format_bytes
from src.utils.exceptions import DockerOperationError, CommandExecutionError

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from config.settings import Settings
from pydantic import BaseModel, Field


class DockerAction(BaseModel):
    """Docker operation parsing model"""
    action: str = Field(description="Operation type, such as list_containers, start_container, pull_image, etc.")
    target: Optional[str] = Field(default=None, description="Target container name or image name")


docker_parser = PydanticOutputParser(pydantic_object=DockerAction)

docker_prompt = PromptTemplate(
    template="""You are a professional Docker command parsing expert. Strictly extract structured information from the following natural language instruction:

Available action types (must EXACTLY match one of the following; do NOT invent new types):
- list_containers
- container_status
- start_container
- stop_container
- restart_container
- container_logs
- pull_image
- remove_image
- list_images
- run_image

target: Container name or image name (e.g., web-app or nginx:latest). Use None if not specified in the command.

Command: {command}

{format_instructions}""",
    input_variables=["command"],
    partial_variables={"format_instructions": docker_parser.get_format_instructions()},
)

logger = get_logger(__name__)


class DockerConfig(BaseModel):
    """Docker configuration model"""
    socket_path: str = Field(default_factory=lambda: get_config('docker.socket_path', 'tcp://localhost:2375'))
    timeout: int = Field(default_factory=lambda: get_config('docker.timeout', 30))
    api_version: str = Field(default_factory=lambda: get_config('docker.api_version', 'auto'))


class DockerOpsTool(BaseTool):
    """Docker operations LangChain tool"""

    name: str = "docker_operations"
    description: str = (
        "A tool for executing Docker container and image operations. "
        "Supports starting, stopping, restarting containers, querying container status, "
        "pulling and deleting images, viewing container logs, and more. "
        "Input should be a natural language description, such as 'start container named web-app' "
        "or 'show all container status'."
    )
    args_schema: Optional[BaseModel] = None

    def __init__(self):
        """Initialize Docker tool"""
        super().__init__()

    def _parse_command(self, command: str) -> tuple[str, Optional[str]]:
        """
        Parse Docker command using AI to extract action and target.
        """
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
        chain = docker_prompt | llm | docker_parser
        try:
            parsed = chain.invoke({"command": command})
            return parsed.action, parsed.target
        except Exception as e:
            logger.warning(f"AI command parsing failed: {e}")
            return "unknown", None

    def _run(self, command: str) -> str:
        """
        Execute Docker operation.
        """
        try:
            client = self._get_docker_client()
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
            elif action == "run_image":
                return self._run_image(client, target)
            else:
                return (
                    f"Unsupported Docker operation: {command}. "
                    "Supported operations include: list containers, container status, "
                    "start/stop/restart container, view logs, pull/remove images, list images."
                )

        except Exception as e:
            logger.error(f"Docker operation failed: {e}")
            raise DockerOperationError(f"Docker operation failed: {str(e)}")

    def _get_docker_client(self) -> docker.DockerClient:
        """Get Docker client"""
        config = DockerConfig()
        try:
            client = docker.DockerClient(
                base_url=config.socket_path,
                version=config.api_version,
                timeout=config.timeout
            )
            client.ping()
            return client
        except Exception as e:
            logger.warning(f"Docker API connection failed, falling back to CLI mode: {e}")
            return None

    def _list_containers(self, client) -> str:
        """List all containers"""
        if client:
            try:
                containers = client.containers.list(all=True)
                if not containers:
                    return "No running or stopped containers."

                result = "Docker Containers:\n"
                result += "ID\t\tName\t\tStatus\t\tImage\n"
                result += "-" * 50 + "\n"

                for container in containers:
                    status = container.status
                    name = container.name or "Unnamed"
                    image = container.image.tags[0] if container.image.tags else "Unknown"
                    result += f"{container.short_id}\t{name}\t\t{status}\t\t{image}\n"

                return result
            except Exception as e:
                logger.error(f"Failed to list containers: {e}")
                raise DockerOperationError("Failed to list containers")
        else:
            return self._run_docker_command(
                "docker ps -a --format 'table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}'"
            )

    def _get_container_status(self, client, container_name: str) -> str:
        """Get container status"""
        if not container_name:
            return "Please specify a container name."

        if client:
            try:
                container = client.containers.get(container_name)
                info = container.attrs
                status = info['State']['Status']
                uptime = format_duration(int(info['State']['StartedAt']) / 1000000000) \
                    if 'StartedAt' in info['State'] else "Unknown"

                result = f"Status of container '{container_name}':\n"
                result += f"Status: {status}\n"
                result += f"Uptime: {uptime}\n"
                result += f"Image: {info['Config']['Image']}\n"
                result += f"Port Bindings: {info['HostConfig']['PortBindings']}\n"

                return result
            except docker.errors.NotFound:
                return f"Container '{container_name}' not found."
            except Exception as e:
                logger.error(f"Failed to get container status: {e}")
                raise DockerOperationError(f"Failed to get status for '{container_name}'")
        else:
            return self._run_docker_command(
                f"docker inspect {container_name} --format='{{json .State}}'"
            )

    def _run_image(self, client, image_name: str, container_name: Optional[str] = None) -> str:
        """Run a container once from an image"""
        if not image_name:
            return "Please specify an image name."

        if client:
            try:
                container = client.containers.run(image_name, name=container_name, detach=False)
                return f"✅ Image '{image_name}' ran successfully!"
            except docker.errors.ImageNotFound:
                return f"Image '{image_name}' not found. Please pull it first."
            except Exception as e:
                logger.error(f"Failed to run image: {e}")
                raise DockerOperationError(f"Failed to run '{image_name}': {str(e)}")
        else:
            result = self._run_docker_command(f"run {image_name}")
            if result[0] == 0:
                return f"✅ Image '{image_name}' ran successfully!"
            else:
                return f"❌ Failed to run '{image_name}': {result[2]}"

    def _start_container(self, client, container_name: str) -> str:
        """Start a container"""
        if not container_name:
            return "Please specify a container name."

        if client:
            try:
                container = client.containers.get(container_name)
                if container.status == 'running':
                    return f"Container '{container_name}' is already running."

                container.start()
                return f"✅ Container '{container_name}' started!\nID: {container.id[:12]}\nStatus: Running"
            except docker.errors.NotFound:
                return f"Container '{container_name}' not found."
            except Exception as e:
                logger.error(f"Failed to start container: {e}")
                raise DockerOperationError(f"Failed to start '{container_name}': {str(e)}")
        else:
            result = self._run_docker_command(f"docker start {container_name}")
            return (
                f"✅ Container '{container_name}' started!"
                if result[0] == 0 else f"❌ Failed to start container: {result[2]}"
            )

    def _stop_container(self, client, container_name: str) -> str:
        """Stop a container"""
        if not container_name:
            return "Please specify a container name."

        if client:
            try:
                container = client.containers.get(container_name)
                if container.status != 'running':
                    return f"Container '{container_name}' is not running."

                container.stop()
                return f"✅ Container '{container_name}' stopped.\nID: {container.id[:12]}"
            except docker.errors.NotFound:
                return f"Container '{container_name}' not found."
            except Exception as e:
                logger.error(f"Failed to stop container: {e}")
                raise DockerOperationError(f"Failed to stop '{container_name}': {str(e)}")
        else:
            result = self._run_docker_command(f"docker stop {container_name}")
            return (
                f"✅ Container '{container_name}' stopped."
                if result[0] == 0 else f"❌ Failed to stop container: {result[2]}"
            )

    def _restart_container(self, client, container_name: str) -> str:
        """Restart a container"""
        if not container_name:
            return "Please specify a container name."

        if client:
            try:
                container = client.containers.get(container_name)
                container.restart()
                return f"✅ Container '{container_name}' restarted.\nID: {container.id[:12]}\nStatus: Running"
            except docker.errors.NotFound:
                return f"Container '{container_name}' not found."
            except Exception as e:
                logger.error(f"Failed to restart container: {e}")
                raise DockerOperationError(f"Failed to restart '{container_name}': {str(e)}")
        else:
            result = self._run_docker_command(f"docker restart {container_name}")
            return (
                f"✅ Container '{container_name}' restarted."
                if result[0] == 0 else f"❌ Failed to restart container: {result[2]}"
            )

    def _get_container_logs(self, client, container_name: str, lines: int = 50) -> str:
        """Get container logs"""
        if not container_name:
            return "Please specify a container name."

        if client:
            try:
                container = client.containers.get(container_name)
                logs = container.logs(tail=lines, stream=False).decode('utf-8')

                if not logs.strip():
                    return f"Container '{container_name}' has no logs."

                if len(logs) > 2000:
                    logs = logs[-2000:] + "\n... (truncated)"

                return f"Latest {lines} lines of logs for '{container_name}':\n\n{logs}"
            except docker.errors.NotFound:
                return f"Container '{container_name}' not found."
            except Exception as e:
                logger.error(f"Failed to fetch logs: {e}")
                raise DockerOperationError(f"Failed to fetch logs for '{container_name}'")
        else:
            result = self._run_docker_command(f"docker logs --tail {lines} {container_name}")
            return (
                f"Latest {lines} lines of logs:\n\n{result[1]}"
                if result[0] == 0 else f"❌ Failed to get logs: {result[2]}"
            )

    def _pull_image(self, client, image_name: str) -> str:
        """Pull Docker image"""
        if not image_name:
            return "Please specify an image name (e.g., nginx:latest)."

        if client:
            try:
                result = client.images.pull(image_name)
                return f"✅ Image '{image_name}' pulled!\nImage ID: {result.id}"
            except Exception as e:
                logger.error(f"Failed to pull image: {e}")
                raise DockerOperationError(f"Failed to pull '{image_name}': {str(e)}")
        else:
            result = self._run_docker_command(f"docker pull {image_name}")
            return (
                f"✅ Image '{image_name}' pulled!"
                if result[0] == 0 else f"❌ Failed to pull image: {result[2]}"
            )

    def _remove_image(self, client, image_name: str) -> str:
        """Remove Docker image"""
        if not image_name:
            return "Please specify an image name."

        if client:
            try:
                client.images.remove(image_name, force=True)
                return f"✅ Image '{image_name}' removed."
            except Exception as e:
                logger.error(f"Failed to remove image: {e}")
                raise DockerOperationError(f"Failed to remove '{image_name}': {str(e)}")
        else:
            result = self._run_docker_command(f"docker rmi -f {image_name}")
            return (
                f"✅ Image '{image_name}' removed."
                if result[0] == 0 else f"❌ Failed to remove image: {result[2]}"
            )

    def _list_images(self, client) -> str:
        """List all Docker images"""
        if client:
            try:
                images = client.images.list(all=True)
                if not images:
                    return "No Docker images available."

                result = "Docker Images:\n"
                result += "ID\t\tRepository\tTag\tSize\n"
                result += "-" * 40 + "\n"

                for image in images:
                    repo_tags = image.tags if image.tags else ["<none>:<none>"]
                    size = format_bytes(image.attrs['Size'])
                    repo, tag = repo_tags[0].split(":")
                    result += f"{image.short_id}\t{repo}\t{tag}\t{size}\n"

                return result
            except Exception as e:
                logger.error(f"Failed to list images: {e}")
                raise DockerOperationError("Failed to list images")
        else:
            return self._run_docker_command(
                "docker images --format 'table {{.ID}}\t{{.Repository}}\t{{.Tag}}\t{{.Size}}'"
            )

    def _run_docker_command(self, cmd: str) -> tuple:
        """Run Docker command (fallback CLI mode)"""
        try:
            return run_command(f"docker {cmd}", timeout=60)
        except Exception as e:
            raise CommandExecutionError(f"Docker command failed: {str(e)}", f"docker {cmd}")


if __name__ == "__main__":
    try:
        tool = DockerOpsTool()
        print("Testing Docker tool:")
        print(tool._list_containers(None))
    except Exception as e:
        print(f"Test failed: {e}")
