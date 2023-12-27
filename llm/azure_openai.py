import asyncio
import logging
import os
import random
import sys
from typing import List, Tuple

import httpx
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from func_timeout import FunctionTimedOut, func_timeout
from openai import AsyncAzureOpenAI, AzureOpenAI

from utils.token_calculator import num_tokens_from_messages

CONFIG = yaml.safe_load(open("config/config.yaml", "r"))

# httpx timeout settings
CONNECTION_TIMEOUT = 2.0
READ_TIMEOUT = 5.0
WRITE_TIMEOUT = 20.0
POOL_TIMEOUT = 10.0
HTTP_TIMEOUT = httpx.Timeout(
    connect=CONNECTION_TIMEOUT,
    read=READ_TIMEOUT,
    write=WRITE_TIMEOUT,
    pool=POOL_TIMEOUT,
)

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
        __test_regions_endpoints(self, available_regions): Asynchronously tests the endpoints of the available regions.
        __test_endpoint(self, region, api_base): Asynchronously tests a specific endpoint.
        __get_regions_for_model(self, model): Returns the regions where a specific model is available.
        __get_chat_completion(self, is_async, messages, model, timeout, stream, **kwargs): Asynchronously retrieves chat completions from the OpenAI service.
        aget_chat_completion(self, messages, model, timeout, stream, **kwargs): Asynchronously retrieves chat completions from the OpenAI service.
        get_chat_completion(self, messages, model, timeout, stream, **kwargs): Synchronously retrieves chat completions from the OpenAI service.
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

    async def __test_regions_endpoints(self, available_regions: List[str]):
        tasks = [
            self.__test_endpoint(region, self.regions[region]["api_base"])
            for region in available_regions
        ]
        results = await asyncio.gather(*tasks)
        return [region for region, result in zip(available_regions, results) if result]

    async def __test_endpoint(self, region: str, api_base: str):
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

    def __get_regions_for_model(self, model: str):
        return [
            region
            for region, details in self.regions.items()
            if model in details["available_models"]
        ]

    async def aget_chat_completion(
        self,
        messages: List[str],
        model: str = "gpt-35-turbo",
        timeout: float = 30.0,
        stream: bool = True,
        **kwargs,
    ) -> Tuple[str, int]:
        """
        Asynchronously retrieves chat completions from the AsyncAzureOpenAI service.

        Args:
            messages (List[str]): The list of messages to be completed.
            model (str, optional): The model to be used for the completion. Defaults to "gpt-35-turbo".
            timeout (float, optional): The timeout for the request. Defaults to 30.0.
            stream (bool, optional): Whether the response should be streamed. Defaults to True.
            **kwargs: Additional parameters for the chat completion.

        Returns:
            Tuple[str, int]: The completed chat and the number of tokens used.
        """
        parameters = self.parameters.copy()
        parameters.update(kwargs)

        # get available regions for the model
        available_regions = self.__get_regions_for_model(model)
        # use random order to reduce the chance of hitting the same region
        random.shuffle(available_regions)

        # test the endpoints of the available regions
        success_regions = await self.__test_regions_endpoints(available_regions)
        # times 3 to retry the failed regions
        success_regions = success_regions * 3

        for region in success_regions:
            api_base = self.regions[region]["api_base"]

            client = AsyncAzureOpenAI(
                azure_endpoint=api_base,
                api_key=self.regions[region]["api_key"],
                api_version=self.api_version,
            )

            logger.debug(f"Start using endpoint {api_base} | {model}")
            try:
                try:
                    response = await asyncio.wait_for(
                        client.with_options(
                            max_retries=1,
                            timeout=HTTP_TIMEOUT,
                        ).chat.completions.create(
                            model=model,
                            messages=messages,
                            stream=stream,
                            **parameters,
                        ),
                        timeout,
                    )
                except asyncio.TimeoutError as e:
                    logger.error(
                        f"Timeout error using endpoint {api_base} | {model}: {e}"
                    )
                    continue

                if not stream:
                    content = response.choices[0].message.content
                    tokens = response.usage.total_tokens
                    yield content, tokens
                    return

                tokens = num_tokens_from_messages(messages, model=model)
                async for chunk in response:
                    try:
                        chunk_message = chunk.choices[0].delta.content
                        if chunk_message is not None:
                            tokens += 1
                            yield chunk_message, tokens
                    except:
                        pass
                return
            except Exception as e:
                logger.error(f"Error using endpoint {api_base} | {model}: {e}")

        raise RuntimeError("All endpoints failed")

    def get_chat_completion(
        self,
        messages: List[str],
        model: str = "gpt-35-turbo",
        timeout: float = 30.0,
        stream: bool = True,
        **kwargs,
    ) -> Tuple[str, int]:
        """
        Synchronously retrieves chat completions from the AzureOpenAI service.

        Args:
            messages (List[str]): The list of messages to be completed.
            model (str, optional): The model to be used for the completion. Defaults to "gpt-35-turbo".
            timeout (float, optional): The timeout for the request. Defaults to 30.0.
            stream (bool, optional): Whether the response should be streamed. Defaults to True.
            **kwargs: Additional parameters for the chat completion.

        Returns:
            Tuple[str, int]: The completed chat and the number of tokens used.
        """
        parameters = self.parameters.copy()
        parameters.update(kwargs)

        # get available regions for the model
        available_regions = self.__get_regions_for_model(model)
        # use random order to reduce the chance of hitting the same region
        random.shuffle(available_regions)

        # test the endpoints of the available regions
        success_regions = asyncio.run(self.__test_regions_endpoints(available_regions))
        # times 3 to retry the failed regions
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
                            max_retries=1, timeout=HTTP_TIMEOUT
                        ).chat.completions.create,
                        kwargs={
                            "model": model,
                            "messages": messages,
                            "stream": stream,
                            **parameters,
                        },
                    )
                except FunctionTimedOut as e:
                    logger.error(
                        f"Timeout error using endpoint {api_base} | {model}: {e}"
                    )
                    continue

                if not stream:
                    content = response.choices[0].message.content
                    tokens = response.usage.total_tokens
                    yield content, tokens
                    return

                tokens = num_tokens_from_messages(messages, model=model)
                for chunk in response:
                    try:
                        chunk_message = chunk.choices[0].delta.content
                        if chunk_message is not None:
                            tokens += 1
                            yield chunk_message, tokens
                    except:
                        pass
                return
            except Exception as e:
                logger.error(f"Error using endpoint {api_base} | {model}: {e}")

        raise RuntimeError("All endpoints failed")
