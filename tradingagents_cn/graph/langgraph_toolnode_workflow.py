"""正式 LangGraph + ToolNode 工具调用工作流。

这个文件替代早期手写工具调用循环：

    模型返回 tool_calls
    Python 手动解析 arguments
    Python 手动执行工具

现在改成 LangGraph 的正式结构：

    agent 节点
        调用大模型，判断是否需要工具

    tools 节点
        LangGraph ToolNode 自动执行工具

    条件边
        如果 agent 返回 tool_calls，就进入 tools；
        否则结束。

    tools -> agent
        工具结果回到 agent，让模型生成最终回答。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents_cn.graph.checkpointing import build_thread_config, create_memory_checkpointer
from tradingagents_cn.llm.factory import create_chat_client
from tradingagents_cn.tools.registry import get_trading_tools


SYSTEM_PROMPT = """你是 A 股交易研究助手。

你必须遵守以下规则：

1. 用户询问行情、新闻、基本面、情绪面时，应优先调用工具获取数据。
2. 工具返回的是原材料，你需要基于工具结果做归纳，不要机械复述。
3. 不要编造工具没有返回的具体数字、新闻、财务数据或社区观点。
4. 如果工具结果缺失，要明确说明数据缺口。
5. 本系统用于个人投资研究辅助，最终决策由使用者自行确认。
"""


@dataclass
class LangGraphToolNodeResult:
    """LangGraph ToolNode 工作流结果。"""

    final_answer: str
    messages: list[BaseMessage]
    thread_id: str


def build_tools_schema(tools: Iterable[BaseTool]) -> list[dict[str, Any]]:
    """把 LangChain 工具转换成 OpenAI 兼容 tools schema。"""
    schemas: list[dict[str, Any]] = []
    for tool_item in tools:
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


def langchain_messages_to_payload(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """把 LangChain Message 转成 OpenAI 兼容 dict。"""
    payload: list[dict[str, Any]] = []
    for message in messages:
        if message.type == "system":
            payload.append({"role": "system", "content": message.content})
        elif message.type == "human":
            payload.append({"role": "user", "content": message.content})
        elif message.type == "ai":
            item: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call.get("args", {}), ensure_ascii=False),
                        },
                    }
                    for call in tool_calls
                ]
            payload.append(item)
        elif message.type == "tool":
            payload.append(
                {
                    "role": "tool",
                    "tool_call_id": getattr(message, "tool_call_id", ""),
                    "content": message.content,
                }
            )
    return payload


def assistant_dict_to_ai_message(message: dict[str, Any]) -> AIMessage:
    """把模型返回的 assistant dict 转成 LangChain AIMessage。"""
    tool_calls = []
    for raw_call in message.get("tool_calls") or []:
        function_info = raw_call.get("function", {})
        raw_args = function_info.get("arguments") or "{}"
        if isinstance(raw_args, str):
            args = json.loads(raw_args)
        else:
            args = raw_args
        tool_calls.append(
            {
                "name": function_info.get("name"),
                "args": args,
                "id": raw_call.get("id"),
                "type": "tool_call",
            }
        )

    return AIMessage(
        content=message.get("content") or "",
        tool_calls=tool_calls,
    )


def should_continue(state: MessagesState) -> str:
    """根据最后一条 AIMessage 判断下一步走 tools 还是结束。"""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


def build_langgraph_toolnode_app(
    llm_client: Any | None = None,
    tools: list[BaseTool] | None = None,
    checkpointer: Any | None = None,
):
    """构建正式 LangGraph ToolNode app。"""
    actual_client = llm_client or create_chat_client()
    actual_tools = tools or get_trading_tools()
    tool_schema = build_tools_schema(actual_tools)

    def call_agent(state: MessagesState) -> dict[str, list[AIMessage]]:
        """agent 节点：调用模型，让模型决定是否需要工具。"""
        response = actual_client.chat(
            messages=langchain_messages_to_payload(state["messages"]),
            tools=tool_schema,
            tool_choice="auto",
            temperature=0.2,
        )
        choices = response.get("choices") or []
        if not choices:
            raise ValueError("模型返回结果中没有 choices。")
        raw_message = choices[0].get("message")
        if not isinstance(raw_message, dict):
            raise ValueError("模型返回结果中没有合法 message。")
        return {"messages": [assistant_dict_to_ai_message(raw_message)]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_agent)
    workflow.add_node("tools", ToolNode(actual_tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer or create_memory_checkpointer())


def run_langgraph_toolnode_workflow(
    question: str,
    llm_client: Any | None = None,
    tools: list[BaseTool] | None = None,
    thread_id: str = "default",
    app: Any | None = None,
) -> LangGraphToolNodeResult:
    """运行一次正式 LangGraph ToolNode 工作流。"""
    actual_app = app or build_langgraph_toolnode_app(llm_client=llm_client, tools=tools)
    initial_messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    output = actual_app.invoke(
        {"messages": initial_messages},
        config=build_thread_config(thread_id),
    )
    messages = output["messages"]
    final_message = messages[-1]
    return LangGraphToolNodeResult(
        final_answer=str(final_message.content or ""),
        messages=messages,
        thread_id=thread_id,
    )
