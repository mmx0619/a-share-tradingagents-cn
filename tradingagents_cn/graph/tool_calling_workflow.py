"""正式 Tool Calling 工作流。

这个文件是正式工程里的“编排层”。

它负责把下面几件事串起来：

    用户问题
      -> 发给 DeepSeek，同时告诉模型有哪些工具
      -> 模型返回 tool_calls
      -> Python 执行真实工具
      -> 工具结果作为 tool 消息发回模型
      -> 模型输出最终回答

注意：
    这不是 demo 文件。

    它是后续 A 股版 TradingAgents 的正式工作流基础。
    以后 Market Agent、News Agent、Risk Agent、Trader Agent
    都可以继续在这个工作流基础上拆分和扩展。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tradingagents_cn.llm.deepseek_client import extract_assistant_message
from tradingagents_cn.llm.factory import create_chat_client
from tradingagents_cn.tools.registry import (
    ToolExecutionResult,
    build_openai_compatible_tools_schema,
    execute_tool_call,
)


SYSTEM_PROMPT = """你是 A 股交易研究助手。

你必须遵守以下规则：

1. 当用户询问当前行情、最新价格、涨跌幅、成交额时，优先调用行情工具。
2. 当用户询问最近新闻、消息面、事件影响时，优先调用新闻工具。
3. 如果一个问题同时涉及行情和新闻，可以连续或并行请求多个工具。
4. 工具返回的是原材料，你需要基于工具结果做归纳，而不是机械复述。
5. 回答必须提醒：本系统用于个人投资研究辅助，最终决策由使用者自行确认。
"""


@dataclass
class ToolCallingWorkflowResult:
    """正式工作流的返回结果。

    final_answer:
        大模型基于工具结果生成的最终回答。

    tool_results:
        Python 实际执行过的工具结果。
        调试时可以看这里，确认工具到底有没有被调用。

    messages:
        完整消息链。
        这对你在 Trae/VSCode 里调试特别有用，
        可以逐行看到每一步传给模型的内容。
    """

    final_answer: str
    tool_results: list[ToolExecutionResult]
    messages: list[dict[str, Any]]


def build_force_tool_choice(tool_name: str) -> dict[str, Any]:
    """构造“强制模型调用某个工具”的 tool_choice。

    正常正式流程一般用 auto，让模型自己判断。

    但有些场景我们希望保证工具调用率，例如：
        用户明确问“现在行情怎么样”。

    这时可以传入：
        force_first_tool_name="akshare_realtime_quote"

    工作流第一次请求模型时，就会强制它返回这个工具调用。
    """
    return {
        "type": "function",
        "function": {"name": tool_name},
    }


class StockToolCallingWorkflow:
    """A 股工具调用工作流。

    当前第 50 步支持：
        - DeepSeek；
        - 实时行情工具；
        - 个股新闻工具。

    后续会继续升级成多 Agent 工作流：
        - Market Agent；
        - News Agent；
        - Summary Agent；
        - Risk Agent；
        - Trader Agent。
    """

    def __init__(self, llm_client=None) -> None:
        """创建工作流。

        llm_client:
            默认使用 DeepSeek。
            以后如果增加 OpenAI/Kimi/Gemini，可以把这里抽成统一接口。
        """
        self.llm_client = llm_client or create_chat_client()

    def run(
        self,
        question: str,
        force_first_tool_name: str | None = None,
    ) -> ToolCallingWorkflowResult:
        """执行一次完整的“用户问题 -> 工具 -> 最终回答”流程。

        question:
            用户原始问题，例如：
                002361 现在行情怎么样？最近有什么新闻？

        force_first_tool_name:
            是否强制第一次模型调用某个工具。

            常见值：
                None：让模型自己判断；
                "akshare_realtime_quote"：强制先查行情；
                "akshare_stock_news"：强制先查新闻。

        返回：
            ToolCallingWorkflowResult。
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        tool_choice: str | dict[str, Any] | None = None
        if force_first_tool_name:
            tool_choice = build_force_tool_choice(force_first_tool_name)

        first_response = self.llm_client.chat(
            messages=messages,
            tools=build_openai_compatible_tools_schema(),
            tool_choice=tool_choice,
        )
        assistant_message = extract_assistant_message(first_response)
        messages.append(assistant_message)

        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            # 如果模型认为不需要工具，就直接返回它的自然语言回答。
            # 这不是错误。
            # 例如用户问“什么是 Tool Calling”，就不需要查行情。
            return ToolCallingWorkflowResult(
                final_answer=str(assistant_message.get("content") or ""),
                tool_results=[],
                messages=messages,
            )

        tool_results: list[ToolExecutionResult] = []
        for tool_call in tool_calls:
            result = execute_tool_call(tool_call)
            tool_results.append(result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.content,
                }
            )

        second_response = self.llm_client.chat(messages=messages)
        final_message = extract_assistant_message(second_response)
        final_answer = str(final_message.get("content") or "")
        messages.append(final_message)

        return ToolCallingWorkflowResult(
            final_answer=final_answer,
            tool_results=tool_results,
            messages=messages,
        )
