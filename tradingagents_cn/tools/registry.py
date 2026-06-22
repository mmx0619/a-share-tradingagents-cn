"""正式工程的工具注册表。

这个文件解决一个工程问题：

项目里的工具会越来越多，例如：
    - 实时行情工具；
    - 个股新闻工具；
    - 公告工具；
    - 财报工具；
    - 技术指标工具。

如果每个工作流都自己 import 一堆工具文件，
后面代码会很乱。

所以这里统一提供：

    get_trading_tools()

工作流只需要问注册表：

    当前项目有哪些工具可以给大模型使用？

然后注册表统一返回。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tradingagents_cn.tools.akshare_tools import get_akshare_tools
from tradingagents_cn.tools.announcement_tools import get_announcement_tools
from tradingagents_cn.tools.fundamentals_tools import get_fundamentals_tools
from tradingagents_cn.tools.news_tools import get_news_tools
from tradingagents_cn.tools.sentiment_tools import get_sentiment_tools


@dataclass
class ToolExecutionResult:
    """一次工具调用的执行结果。

    tool_call_id:
        大模型返回的工具调用 ID。
        后面把工具结果发回模型时，必须带上这个 ID。

    tool_name:
        工具名称，例如 akshare_realtime_quote。

    content:
        Python 真正执行工具后得到的文本结果。
    """

    tool_call_id: str
    tool_name: str
    content: str


def get_trading_tools() -> list:
    """返回当前正式工程里所有可用的交易分析工具。

    第 50 步当前包含：
        - akshare_realtime_quote：A 股实时/近实时行情；
        - akshare_stock_news：A 股个股新闻。
        - get_fundamentals：A 股综合基本面；
        - get_balance_sheet：A 股资产负债表；
        - get_cashflow：A 股现金流量表；
        - get_income_statement：A 股利润表。
        - get_stock_sentiment：A 股社区情绪材料。
        - get_stock_announcements_tool：A 股公司公告/信息披露。

    后续新增工具时，优先加到这里。
    这样正式工作流不用关心工具来自哪个文件。
    """
    return [
        *get_akshare_tools(),
        *get_news_tools(),
        *get_announcement_tools(),
        *get_fundamentals_tools(),
        *get_sentiment_tools(),
    ]


def get_tool_map() -> dict[str, Any]:
    """把工具列表转换成按名称查询的字典。

    大模型返回 tool_calls 时，只会告诉程序：

        我要调用哪个工具名；
        参数是什么。

    程序需要根据工具名找到真正的 Python 工具对象。

    例如：
        {
            "akshare_realtime_quote": 工具对象,
            "akshare_stock_news": 工具对象,
        }
    """
    return {
        tool_item.name: tool_item
        for tool_item in get_trading_tools()
    }


def build_openai_compatible_tools_schema() -> list[dict[str, Any]]:
    """把 LangChain 工具转换成 OpenAI/DeepSeek 兼容的 tools schema。

    DeepSeek 的 Tool Calling 接口和 OpenAI 风格很像。

    它需要的格式大致是：

        [
            {
                "type": "function",
                "function": {
                    "name": "工具名",
                    "description": "工具说明",
                    "parameters": {
                        "type": "object",
                        "properties": {...},
                        "required": [...]
                    }
                }
            }
        ]

    LangChain 的 @tool 已经记录了工具名、说明、参数。
    这里负责把 LangChain 工具对象翻译成模型 API 能看懂的格式。
    """
    schemas: list[dict[str, Any]] = []

    for tool_item in get_trading_tools():
        args_schema = getattr(tool_item, "args_schema", None)

        if args_schema is not None and hasattr(args_schema, "model_json_schema"):
            raw_schema = args_schema.model_json_schema()
            properties = raw_schema.get("properties", {})
            required = raw_schema.get("required", list(properties.keys()))
        else:
            properties = getattr(tool_item, "args", {})
            required = list(properties.keys())

        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool_item.name,
                    "description": tool_item.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )

    return schemas


def execute_tool_call(tool_call: dict[str, Any]) -> ToolExecutionResult:
    """执行大模型返回的一个 tool_call。

    重要理解：

    大模型不会真的访问 AKShare。
    大模型只返回类似这样的结构：

        {
            "id": "call_xxx",
            "type": "function",
            "function": {
                "name": "akshare_realtime_quote",
                "arguments": "{\"symbol\":\"002361\"}"
            }
        }

    真正执行工具的是这里的 Python 代码。
    """
    tool_call_id = str(tool_call.get("id", ""))
    function_info = tool_call.get("function", {})
    tool_name = function_info.get("name")
    raw_arguments = function_info.get("arguments", "{}")

    if not tool_call_id:
        raise ValueError("tool_call 缺少 id，无法把工具结果发回模型。")

    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("tool_call 缺少 function.name，无法判断要调用哪个工具。")

    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as error:
            raise ValueError(f"工具参数不是合法 JSON：{raw_arguments}") from error
    elif isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        raise ValueError(f"工具参数类型不支持：{type(raw_arguments).__name__}")

    tool_map = get_tool_map()
    tool_item = tool_map.get(tool_name)
    if tool_item is None:
        raise ValueError(f"模型请求了未注册的工具：{tool_name}")

    content = tool_item.invoke(arguments)
    return ToolExecutionResult(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=str(content),
    )
