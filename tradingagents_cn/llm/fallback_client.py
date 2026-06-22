"""多模型自动兜底客户端。

用户当前希望：
    首选 DeepSeek；
    DeepSeek 不可用时自动切到 Gemini。

这个模块把兜底逻辑放在 LLM 层，而不是散落到每个 Agent 里。
这样 Market / News / Fundamentals / Risk / Trader 等所有节点都会一起生效。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatClientCandidate:
    """一个可被尝试的模型客户端。"""

    provider: str
    client: Any


@dataclass
class FallbackChatClient:
    """按顺序尝试多个聊天模型客户端。

    行为说明：
        1. 默认从第一个客户端开始，例如 deepseek；
        2. 如果该客户端调用失败，切到下一个，例如 gemini；
        3. 一旦切到后面的客户端，本轮 Python 进程后续调用都优先用它；
        4. 如果所有客户端都失败，抛出包含所有错误的 RuntimeError。

    为什么要记住 active_index：
        如果 DeepSeek 已经因为 402 失败，再让每个 Agent 都先试一次 DeepSeek，
        只会制造重复报错和浪费时间。
    """

    candidates: list[ChatClientCandidate]
    active_index: int = 0
    failure_history: list[str] = field(default_factory=list)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """调用当前模型；失败时自动尝试下一个模型。"""
        if not self.candidates:
            raise RuntimeError("没有可用的大模型客户端。")

        errors: list[str] = []
        start_index = min(max(self.active_index, 0), len(self.candidates) - 1)

        for index in range(start_index, len(self.candidates)):
            candidate = self.candidates[index]
            try:
                response = candidate.client.chat(
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                )
                self.active_index = index
                return response
            except Exception as error:
                message = f"{candidate.provider} 调用失败：{error}"
                errors.append(message)
                self.failure_history.append(message)
                self.active_index = min(index + 1, len(self.candidates) - 1)

        raise RuntimeError(
            "所有大模型都调用失败。\n" + "\n".join(f"- {item}" for item in errors)
        )

    @property
    def active_provider(self) -> str:
        """返回当前优先使用的 provider 名称。"""
        if not self.candidates:
            return ""
        index = min(max(self.active_index, 0), len(self.candidates) - 1)
        return self.candidates[index].provider


def build_fallback_chat_client(candidates: list[tuple[str, Any]]) -> FallbackChatClient:
    """根据 provider/client 元组构造 FallbackChatClient。"""
    return FallbackChatClient(
        candidates=[
            ChatClientCandidate(provider=provider, client=client)
            for provider, client in candidates
        ]
    )
