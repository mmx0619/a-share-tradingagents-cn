"""OpenAI 兼容聊天模型客户端。

很多大模型服务商虽然名字不同，但都提供了类似 OpenAI 的
Chat Completions 接口，例如：

    DeepSeek
    OpenAI
    Kimi / Moonshot
    Gemini OpenAI-compatible endpoint

这个文件把公共 HTTP 调用逻辑抽出来，后续切换模型时不用改 Agent 代码。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests

from tradingagents_cn.llm.deepseek_client import should_retry_http_status
from tradingagents_cn.llm.errors import LLMAPIError, build_llm_http_error_message


@dataclass
class OpenAICompatibleChatClient:
    """OpenAI 兼容聊天模型客户端。

    api_key_env:
        从哪个环境变量读取 API Key。
        可以传一个字符串，也可以传多个候选环境变量。
        例如 Gemini 常见变量有 GEMINI_API_KEY 和 GOOGLE_API_KEY。

    model:
        模型名称，例如 gpt-4o-mini、deepseek-chat、moonshot-v1-8k。

    base_url:
        完整 chat completions 地址。
        注意这里是完整地址，不只是域名。
    """

    api_key_env: str | tuple[str, ...]
    model: str
    base_url: str
    timeout: int = 90
    max_retries: int = 3
    retry_sleep_seconds: float = 2.0

    def get_api_key(self) -> str:
        """从环境变量读取 API Key。"""
        env_names = normalize_env_names(self.api_key_env)
        for env_name in env_names:
            api_key = os.environ.get(env_name)
            if api_key:
                return api_key
        joined = " 或 ".join(env_names)
        raise RuntimeError(f"没有读取到 {joined}，请先配置环境变量。")

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """调用聊天模型，并返回原始 JSON。"""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice if tool_choice is not None else "auto"

        return self.post_with_retry(payload)

    def post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """发送请求，并对网络抖动、429、5xx 做有限重试。"""
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.get_api_key()}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )

                if should_retry_http_status(response.status_code):
                    last_error = RuntimeError(
                        f"模型接口临时 HTTP 错误：{response.status_code}，"
                        f"第 {attempt}/{self.max_retries} 次。"
                    )
                    if attempt < self.max_retries:
                        time.sleep(self.retry_sleep_seconds)
                        continue

                response.raise_for_status()
                return response.json()

            except (requests.Timeout, requests.ConnectionError) as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep_seconds)
                    continue
                break

            except requests.HTTPError as error:
                # API Key 错、模型名错、权限错这类 4xx 问题，重试没有意义。
                raise LLMAPIError(
                    build_llm_http_error_message(self.model, error)
                ) from error

        raise RuntimeError(
            f"模型调用失败，已重试 {self.max_retries} 次。最后错误：{last_error}"
        )


def normalize_env_names(value: str | tuple[str, ...]) -> tuple[str, ...]:
    """把单个或多个环境变量名规范化成 tuple。"""
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value if str(item).strip())
