"""DeepSeek 大模型客户端。

这个文件只负责一件事：

    调用 DeepSeek Chat Completions API。

它不负责：
    - 采集行情；
    - 采集新闻；
    - 执行工具；
    - 做工作流路由。

为什么要单独拆出来？

因为后面你可能还会接：
    - OpenAI；
    - Gemini；
    - Kimi；
    - 其他兼容 OpenAI 接口的模型。

如果把 API 调用代码散落在各个 Agent 里，
以后切换模型会很痛苦。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests

from tradingagents_cn.llm.errors import LLMAPIError, build_llm_http_error_message


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


@dataclass
class DeepSeekChatClient:
    """DeepSeek 聊天模型客户端。

    api_key_env:
        从哪个环境变量读取 API Key。
        默认读取 DEEPSEEK_API_KEY。

    model:
        使用哪个 DeepSeek 模型。
        如果不传，就优先读取 DEEPSEEK_MODEL；
        环境变量也没有时，使用 deepseek-chat。

    base_url:
        DeepSeek Chat Completions API 地址。
    """

    api_key_env: str = "DEEPSEEK_API_KEY"
    model: str | None = None
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    timeout: int = 90
    max_retries: int = 3
    retry_sleep_seconds: float = 2.0

    def get_api_key(self) -> str:
        """从环境变量读取 API Key。

        注意：
            API Key 不写进代码。
            也不写进日志。
        """
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"没有读取到 {self.api_key_env}，请先配置环境变量。")
        return api_key

    def get_model(self) -> str:
        """返回本次调用使用的模型名称。"""
        if self.model:
            return self.model
        return os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """调用 DeepSeek，并返回原始 JSON。

        messages:
            对话消息列表。

        tools:
            可选工具 schema。
            如果传入，模型可以返回 tool_calls。

        tool_choice:
            控制模型是否调用工具。

            常用值：
                None：如果有 tools，就默认 auto；
                "auto"：让模型自己判断；
                具体 dict：强制调用某个工具。

        temperature:
            随机性。
            股票分析场景一般不希望太发散，所以默认 0.2。
        """
        payload: dict[str, Any] = {
            "model": self.get_model(),
            "messages": messages,
            "temperature": temperature,
        }

        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice if tool_choice is not None else "auto"

        return self.post_with_retry(payload)

    def post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """发送请求，并对临时网络问题做有限重试。

        哪些情况会重试：
            - 网络连接错误；
            - 请求超时；
            - 服务端 5xx 错误；
            - 429 限流错误。

        哪些情况不会重试：
            - 400 请求参数错误；
            - 401 API Key 错误；
            - 403 权限错误；
            - 404 地址错误；
            - 其他明显不是“再试一次就能好”的 4xx 错误。

        为什么这样设计？
            网络抖动、服务端临时错误，重试有意义。
            API Key 错、模型名错、请求格式错，循环重试只会浪费时间和额度。
        """
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
                        f"DeepSeek 临时 HTTP 错误：{response.status_code}，"
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
                # 4xx 这类 API Key、参数、权限问题，不在这里吞掉。
                # 让调用方看到明确错误，方便修配置。
                raise LLMAPIError(build_llm_http_error_message("DeepSeek", error)) from error

        raise RuntimeError(
            f"DeepSeek 调用失败，已重试 {self.max_retries} 次。"
            f"最后错误：{last_error}"
        )


def should_retry_http_status(status_code: int) -> bool:
    """判断 HTTP 状态码是否值得重试。"""
    if status_code == 429:
        return True

    if 500 <= status_code <= 599:
        return True

    return False


def extract_assistant_message(response_json: dict[str, Any]) -> dict[str, Any]:
    """从模型返回 JSON 中取出 assistant message。

    DeepSeek 返回结构通常是：

        {
            "choices": [
                {
                    "message": {...}
                }
            ]
        }

    工作流真正关心的是 choices[0].message。
    """
    choices = response_json.get("choices")
    if not choices:
        raise ValueError("模型返回结果中没有 choices。")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("模型返回结果中没有合法的 assistant message。")

    return message
