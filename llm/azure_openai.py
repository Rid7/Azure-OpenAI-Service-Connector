import asyncio
import logging
import random
from typing import List, Tuple

import httpx
import yaml

from func_timeout import func_timeout, FunctionTimedOut
from openai import AzureOpenAI
from utils.token_calculator import num_tokens_from_messages

CONFIG = yaml.safe_load(open("config/config.yaml", "r"))
logger = logging.getLogger(__name__)


class OpenAIService:
    """
    A class that provides methods for interacting with the Azure OpenAI service.

    Attributes:
        regions (dict): A dictionary containing the regions and their details.
        api_type (str): The type of API used.
        api_version (str): The version of the API used.
        parameters (dict): A dictionary containing the parameters for the chat completion.

    Methods:
        __init__(self, config=CONFIG["AZURE_OPENAI"]): Initializes the OpenAIService object.
        __test_regions_endpoints(self, available_regions): Tests the endpoints of the available regions asynchronously.
        __test_endpoint(self, region, api_base): Tests a specific endpoint.
        __get_regions_for_model(self, model): Returns the regions where a specific model is available.
        get_chat_completion(self, messages, model, timeout, stream, **kwargs): Retrieves chat completions from the OpenAI service.
    """

    def __init__(self, config=CONFIG["AZURE_OPENAI"]):
        self.regions = config["regions"]
        self.api_type = config["api_type"]
        self.api_version = config["api_version"]
        self.parameters = {
            "temperature": 0,
            "top_p": 0.7,
            "n": 1,
            "seed": 42,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }

    async def __test_regions_endpoints(self, available_regions):
        tasks = [
            self.__test_endpoint(region, self.regions[region]["api_base"])
            for region in available_regions
        ]
        results = await asyncio.gather(*tasks)
        return [region for region, result in zip(available_regions, results) if result]

    async def __test_endpoint(self, region, api_base):
        """
        Test the endpoint for a specific region and API base by pre-ping.

        Args:
            region (str): The region of the endpoint.
            api_base (str): The base URL of the API.

        Returns:
            bool: True if the endpoint is reachable, False otherwise.
        """
        api_base = api_base.replace("https://", "").replace("http://", "").strip("/")
        cmd = ["ping", "-W", "1", "-c", "1", api_base]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                std_out, std_err = await asyncio.wait_for(
                    process.communicate(), timeout=2
                )
                std_out = std_out.decode()
                std_err = std_err.decode()
            except asyncio.TimeoutError:
                process.terminate()
                await process.wait()

            if "100% packet loss" not in std_out and std_err == "":
                logger.debug(f"The endpoint for region {region} is working fine")
                return True
            else:
                logger.debug(f"The endpoint for region {region} is not reachable")
                return False
        except Exception as e:
            logger.debug(
                f"The endpoint for region {region} is not reachable, error: {e}"
            )
            return False

    def __get_regions_for_model(self, model):
        return [
            region
            for region, details in self.regions.items()
            if model in details["available_models"]
        ]

    def get_chat_completion(
        self,
        messages: List[str],
        model: str = "gpt-35-turbo",
        timeout: float = 30.0,
        stream: bool = True,
        **kwargs,
    ) -> Tuple[str, int]:
        """
        获取聊天完成结果。

        Args:
            messages (List[str]): 聊天消息列表。
            model (str, optional): 模型名称。默认为"gpt-35-turbo"。
            timeout (float, optional): 超时时间（秒）。
            stream (bool, optional): 是否使用流式处理。默认为True。
            **kwargs: 其他可选参数。

        Returns:
            Tuple[str, int]: 聊天完成的内容和使用的token数。

        Raises:
            Exception: 所有的节点都失败。
        """
        http_timeout = httpx.Timeout(timeout)
        parameters = self.parameters
        parameters.update(kwargs)

        # 获取当前模型的可用地区，并打乱顺序(随机节点顺序，实现最简单的负载均衡策略)
        available_regions = self.__get_regions_for_model(model)
        random.shuffle(available_regions)

        # 测试各个地区的endpoint是否可用
        success_regions = asyncio.run(self.__test_regions_endpoints(available_regions))

        # 扩充列表三倍，达到重试三次的效果
        success_regions = success_regions * 3

        for region in success_regions:
            api_base = self.regions[region]["api_base"]

            client = AzureOpenAI(
                azure_endpoint=api_base,
                api_key=self.regions[region]["api_key"],
                api_version=self.api_version,
            )

            logger.debug(f"Start using endpoint {api_base} | {model}")
            try:
                try:
                    response = func_timeout(
                        timeout,
                        client.with_options(
                            max_retries=1, timeout=http_timeout
                        ).chat.completions.create,
                        kwargs={
                            "model": model,
                            "messages": messages,
                            "stream": stream,
                            **parameters,
                        },
                    )
                except FunctionTimedOut as e:
                    raise Exception("Function timed out") from e

                if not stream:
                    content = response.choices[0].message.content
                    tokens = response.usage.total_tokens
                    return content, tokens

                collected_messages = []
                tokens = num_tokens_from_messages(messages, model=model)
                for chunk in response:
                    try:
                        chunk_message = chunk.choices[0].delta.content
                        if chunk_message is not None:
                            collected_messages.append(chunk_message)
                            tokens += 1
                    except:
                        pass

                content = "".join(collected_messages)
                return content, tokens

            except Exception as e:
                logger.error(f"Error using endpoint {api_base} | {model}: {e}")

        raise Exception("All endpoints failed")
