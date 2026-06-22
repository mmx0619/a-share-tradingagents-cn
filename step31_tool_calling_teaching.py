"""第 31 步：单文件看懂 Tool Calling。

你之前问过：

大模型的输入是一段文字，
输出也是一段文字，
那 Tool Calling 到底是什么？

可以这样理解：

普通聊天：

用户问题
  ↓
大模型直接回答一段文字

Tool Calling：

用户问题
  ↓
大模型不直接回答
  ↓
大模型先返回“我要调用哪个工具，以及参数是什么”
  ↓
程序根据这个工具名和参数，真正执行 Python 函数
  ↓
程序拿到工具结果
  ↓
再把工具结果交给大模型
  ↓
大模型基于工具结果生成最终回答

注意：

真正执行工具的不是大模型。
大模型只是“提出工具调用请求”。
Python 程序才是真正调用工具的人。

本文件用单文件演示：

1. 模型如何返回工具调用请求。
2. 程序如何校验工具名和参数。
3. 程序如何执行对应工具函数。
4. 工具结果如何进入最终回答。
5. 工具名不合法时如何兜底。

为了学习清楚：

- 不联网。
- 不调用真实大模型。
- 不导入前面复杂模块。
- 用 mock_model_decide_tool() 模拟模型决定调用工具。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ToolName = Literal["get_realtime_quote", "get_stock_news", "fallback"]


@dataclass
class ToolCall:
    """模型返回的工具调用请求。

    字段说明：

    - tool_name：模型想调用哪个工具。
    - arguments：工具参数。
    - reason：模型为什么想调用这个工具。

    真实 OpenAI / DeepSeek / Kimi 的 Tool Calling，
    本质上也是返回类似这样的结构。

    区别只是：
    真实 API 会把它包装在官方协议字段里。
    这里为了学习，把它简化成 dataclass。
    """

    tool_name: ToolName
    arguments: dict[str, Any]
    reason: str


@dataclass
class ToolResult:
    """工具执行结果。

    字段说明：

    - tool_name：实际执行的工具名。
    - success：工具是否执行成功。
    - content：工具返回的文本内容。
    """

    tool_name: ToolName
    success: bool
    content: str


def get_realtime_quote(symbol: str) -> str:
    """模拟实时行情工具。

    真实项目里，这个工具会调用 AKShare、东方财富或新浪。

    这里不联网，
    只返回一段模拟行情。
    """
    return f"{symbol} 模拟实时行情：最新价 16.10，涨跌幅 1.07%。"


def get_stock_news(symbol: str, max_items: int = 3) -> str:
    """模拟新闻工具。

    真实项目里，这个工具会获取东方财富新闻等数据。

    这里不联网，
    只返回几条模拟新闻事件。
    """
    return (
        f"{symbol} 模拟新闻：最近 {max_items} 条事件包括龙虎榜、"
        "高换手、短线资金博弈。"
    )


def mock_model_decide_tool(user_question: str) -> ToolCall:
    """模拟模型决定调用哪个工具。

    真实 Tool Calling 中，
    这一段是大模型完成的。

    用户问：
    “002361 现在行情怎么样？”

    模型可能不直接回答，
    而是返回：

    {
        "tool_name": "get_realtime_quote",
        "arguments": {"symbol": "002361"}
    }

    这里用简单规则模拟模型判断。
    """
    if "新闻" in user_question or "消息" in user_question:
        return ToolCall(
            tool_name="get_stock_news",
            arguments={"symbol": "002361", "max_items": 3},
            reason="用户询问新闻或消息面，需要调用新闻工具。",
        )

    if "行情" in user_question or "现在" in user_question or "价格" in user_question:
        return ToolCall(
            tool_name="get_realtime_quote",
            arguments={"symbol": "002361"},
            reason="用户询问当前行情，需要调用实时行情工具。",
        )

    return ToolCall(
        tool_name="fallback",
        arguments={},
        reason="无法判断用户需要哪个工具，进入兜底。",
    )


def mock_model_bad_tool(user_question: str) -> ToolCall:
    """模拟模型返回了一个不存在的工具名。

    这用于演示工具名校验。

    真实工程里，模型有可能返回：

    - 拼错的工具名
    - 不存在的工具名
    - 参数不完整

    所以程序不能盲信模型。
    """
    return ToolCall(
        tool_name="fallback",
        arguments={"symbol": "002361"},
        reason="教学版：用 fallback 代表模型工具选择异常。",
    )


def execute_tool_call(tool_call: ToolCall) -> ToolResult:
    """执行工具调用。

    这是 Tool Calling 里最关键的一步。

    大模型只是返回：

    我要调用 get_realtime_quote，参数是 {"symbol": "002361"}

    但真正执行：

    get_realtime_quote("002361")

    的是 Python 程序，也就是这个函数。
    """
    if tool_call.tool_name == "get_realtime_quote":
        symbol = tool_call.arguments.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            return ToolResult(
                tool_name="fallback",
                success=False,
                content="实时行情工具调用失败：缺少合法 symbol 参数。",
            )

        content = get_realtime_quote(symbol)
        return ToolResult(
            tool_name="get_realtime_quote",
            success=True,
            content=content,
        )

    if tool_call.tool_name == "get_stock_news":
        symbol = tool_call.arguments.get("symbol")
        max_items = tool_call.arguments.get("max_items", 3)

        if not isinstance(symbol, str) or not symbol:
            return ToolResult(
                tool_name="fallback",
                success=False,
                content="新闻工具调用失败：缺少合法 symbol 参数。",
            )

        if not isinstance(max_items, int) or max_items <= 0:
            return ToolResult(
                tool_name="fallback",
                success=False,
                content="新闻工具调用失败：max_items 必须是正整数。",
            )

        content = get_stock_news(symbol, max_items=max_items)
        return ToolResult(
            tool_name="get_stock_news",
            success=True,
            content=content,
        )

    return ToolResult(
        tool_name="fallback",
        success=False,
        content=f"工具调用失败：不支持的工具名 {tool_call.tool_name}。",
    )


def mock_model_final_answer(
    user_question: str,
    tool_call: ToolCall,
    tool_result: ToolResult,
) -> str:
    """模拟模型基于工具结果生成最终回答。

    Tool Calling 通常不是调用工具就结束。

    更完整的链路是：

    1. 用户提问。
    2. 模型请求调用工具。
    3. 程序执行工具。
    4. 程序把工具结果交回模型。
    5. 模型基于工具结果组织自然语言回答。

    这里用普通函数模拟最后一步。
    """
    if not tool_result.success:
        return (
            "我无法完成这次工具查询，系统已进入兜底处理。"
            f"原因：{tool_result.content}"
        )

    return f"""用户问题：
{user_question}

模型选择的工具：
{tool_call.tool_name}

选择原因：
{tool_call.reason}

工具返回结果：
{tool_result.content}

最终回答：
根据工具返回结果，可以看到当前信息如下：{tool_result.content}
这只是工具查询结果，不构成投资建议。
"""


def run_tool_calling_demo(user_question: str) -> str:
    """运行一次完整 Tool Calling 演示。"""
    tool_call = mock_model_decide_tool(user_question)
    tool_result = execute_tool_call(tool_call)
    final_answer = mock_model_final_answer(
        user_question=user_question,
        tool_call=tool_call,
        tool_result=tool_result,
    )

    return f"""======== 用户问题 ========
{user_question}

======== 第一步：模型返回工具调用请求 ========
tool_name = {tool_call.tool_name}
arguments = {tool_call.arguments}
reason = {tool_call.reason}

======== 第二步：程序执行工具 ========
success = {tool_result.success}
tool_name = {tool_result.tool_name}
content = {tool_result.content}

======== 第三步：模型基于工具结果生成最终回答 ========
{final_answer}
"""


def demo_tool_calling() -> None:
    """演示三种问题。"""
    print(run_tool_calling_demo("002361 现在行情怎么样？"))
    print(run_tool_calling_demo("002361 最近有什么新闻？"))
    print(run_tool_calling_demo("002361 适合长期持有吗？"))


if __name__ == "__main__":
    demo_tool_calling()
