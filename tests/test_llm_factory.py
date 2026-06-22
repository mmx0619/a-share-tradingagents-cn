import unittest
from unittest.mock import patch

import requests

from tradingagents_cn.llm.factory import (
    DEFAULT_MODELS,
    create_chat_client,
    get_fallback_providers_from_env,
    normalize_provider,
)
from tradingagents_cn.llm.deepseek_client import DeepSeekChatClient
from tradingagents_cn.llm.errors import LLMAPIError, build_llm_http_error_message
from tradingagents_cn.llm.fallback_client import FallbackChatClient
from tradingagents_cn.llm.openai_compatible_client import OpenAICompatibleChatClient


class LLMFactoryTest(unittest.TestCase):
    def test_normalize_provider_should_support_alias(self):
        """模型服务商别名应归一化成内部名称。"""
        self.assertEqual(normalize_provider("moonshot"), "kimi")
        self.assertEqual(normalize_provider("google"), "gemini")
        self.assertEqual(normalize_provider("gpt"), "openai")

    def test_create_openai_client_should_not_call_network(self):
        """创建客户端不应该触发网络请求。"""
        client = create_chat_client(provider="openai", model="gpt-test")

        self.assertIsInstance(client, OpenAICompatibleChatClient)
        self.assertEqual(client.model, "gpt-test")
        self.assertIn("openai", DEFAULT_MODELS)

    @patch.dict("os.environ", {}, clear=True)
    def test_default_client_should_use_deepseek_then_gemini_fallback(self):
        """默认不传 provider 时，应创建 DeepSeek -> Gemini 兜底链。"""
        client = create_chat_client()

        self.assertIsInstance(client, FallbackChatClient)
        self.assertEqual(["deepseek", "gemini"], [item.provider for item in client.candidates])

    def test_explicit_gemini_provider_should_create_single_gemini_client(self):
        """明确指定 Gemini 时，不应再包一层默认 fallback。"""
        client = create_chat_client(provider="gemini")

        self.assertIsInstance(client, OpenAICompatibleChatClient)
        self.assertEqual("gemini-2.5-flash", client.model)
        self.assertEqual(("GEMINI_API_KEY", "GOOGLE_API_KEY"), client.api_key_env)
        self.assertIn("/openai/chat/completions", client.base_url)

    @patch.dict(
        "os.environ",
        {"TRADINGAGENTS_CN_LLM_FALLBACK_PROVIDERS": "gemini,deepseek,gemini"},
        clear=True,
    )
    def test_fallback_provider_order_should_be_configurable(self):
        """环境变量应能调整 fallback provider 顺序并自动去重。"""
        self.assertEqual(("gemini", "deepseek"), get_fallback_providers_from_env())

    def test_unknown_provider_should_raise(self):
        """未知服务商应直接报错，避免悄悄用错模型。"""
        with self.assertRaises(ValueError):
            normalize_provider("unknown-provider")

    def test_fallback_client_should_switch_to_gemini_after_deepseek_failure(self):
        """DeepSeek 失败时，应自动切到 Gemini，并记住后续优先用 Gemini。"""

        class BrokenClient:
            def chat(self, **kwargs):
                raise LLMAPIError("DeepSeek API 调用失败：HTTP 402")

        class WorkingClient:
            call_count = 0

            def chat(self, **kwargs):
                self.call_count += 1
                return {"choices": [{"message": {"content": "gemini ok"}}]}

        client = FallbackChatClient(
            candidates=[
                type("Candidate", (), {"provider": "deepseek", "client": BrokenClient()})(),
                type("Candidate", (), {"provider": "gemini", "client": WorkingClient()})(),
            ]
        )

        response = client.chat(messages=[{"role": "user", "content": "hello"}])

        self.assertEqual("gemini ok", response["choices"][0]["message"]["content"])
        self.assertEqual("gemini", client.active_provider)
        self.assertIn("deepseek 调用失败", client.failure_history[0])

    def test_http_402_should_explain_payment_problem(self):
        """HTTP 402 应提示余额、计费或额度问题。"""
        response = requests.Response()
        response.status_code = 402
        response.reason = "Payment Required"
        response._content = b'{"error":"insufficient balance"}'
        error = requests.HTTPError("402 Client Error", response=response)

        message = build_llm_http_error_message("DeepSeek", error)

        self.assertIn("HTTP 402", message)
        self.assertIn("余额", message)
        self.assertIn("计费", message)

    @patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"})
    @patch("tradingagents_cn.llm.deepseek_client.requests.post")
    def test_deepseek_client_should_raise_llm_api_error_for_402(self, mock_post):
        """DeepSeek 402 不应继续冒出 requests traceback。"""
        response = requests.Response()
        response.status_code = 402
        response.reason = "Payment Required"
        response._content = b'{"error":"insufficient balance"}'

        def raise_for_status():
            raise requests.HTTPError("402 Client Error", response=response)

        response.raise_for_status = raise_for_status
        mock_post.return_value = response

        client = DeepSeekChatClient(max_retries=1, retry_sleep_seconds=0)

        with self.assertRaises(LLMAPIError) as context:
            client.chat([{"role": "user", "content": "hello"}])

        self.assertIn("HTTP 402", str(context.exception))
        self.assertIn("余额", str(context.exception))


if __name__ == "__main__":
    unittest.main()
