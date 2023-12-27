import asyncio
import logging
import unittest

from llm.azure_openai import OpenAIService
from concurrent.futures import ThreadPoolExecutor


class TestOpenAIService(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.service = OpenAIService()
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(name)s %(levelname)s %(funcName)s:%(lineno)d | %(message)s",
        )

    def test_get_chat_completion(self):
        messages = [{"role": "user", "content": """Hello, how are you?"""}]
        model = "gpt-35-turbo"
        timeout = 30.0
        stream = True

        def run_test():
            for result in self.service.get_chat_completion(
                messages, model, timeout, stream
            ):
                print(result)
                self.assertIsNotNone(result)

        with ThreadPoolExecutor(max_workers=10) as executor:
            for _ in range(100):
                executor.submit(run_test)

    def test_aget_chat_completion(self):
        messages = [{"role": "user", "content": """Hello, how are you?"""}]
        model = "gpt-35-turbo"
        timeout = 30.0
        stream = True

        async def run_test():
            async for result in self.service.aget_chat_completion(
                messages, model, timeout, stream
            ):
                print(result)
                self.assertIsNotNone(result)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
