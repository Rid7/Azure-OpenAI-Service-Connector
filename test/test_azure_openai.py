import unittest
from llm.azure_openai import OpenAIService


class TestOpenAIService(unittest.TestCase):
    def setUp(self):
        self.service = OpenAIService()

    def test_get_chat_completion(
        self,
    ):
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        model = "gpt-35-turbo"
        timeout = 30.0
        stream = True

        result = self.service.get_chat_completion(messages, model, timeout, stream)
        print(result)
        return result


if __name__ == "__main__":
    unittest.main()
