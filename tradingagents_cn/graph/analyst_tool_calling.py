"""Analyst Agent Tool Calling 执行器。

这个文件把三类分析员统一升级为：

    Analyst Agent
      -> 模型决定调用工具
      -> ToolNode 执行工具
      -> 工具结果回到 Analyst Agent
      -> Analyst Agent 输出最终报告

为什么单独抽出来？
    Market / News / Fundamentals 三个分析员都需要同样的 Tool Calling 机制。
    不应该每个节点重复写一套 LangGraph + ToolNode 代码。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents_cn.graph.checkpointing import build_thread_config, create_memory_checkpointer
from tradingagents_cn.graph.langgraph_toolnode_workflow import (
    assistant_dict_to_ai_message,
    build_tools_schema,
    langchain_messages_to_payload,
)
from tradingagents_cn.graph.display_text import localize_report_text
from tradingagents_cn.graph.progress import emit_progress
from tradingagents_cn.llm.factory import create_chat_client


DEFAULT_ANALYST_RECURSION_LIMIT = 30
DEFAULT_ANALYST_MAX_TOOL_ROUNDS = 4


@dataclass
class AnalystToolCallingResult:
    """Analyst Tool Calling 运行结果。"""

    report: str
    messages: list[BaseMessage]
    tool_call_count: int
    tool_trace: list[dict[str, Any]]


def build_force_tool_choice(tool_name: str) -> dict[str, Any]:
    """构造强制模型调用指定工具的 tool_choice。"""
    return {
        "type": "function",
        "function": {
            "name": tool_name,
        },
    }


def should_continue_tools(state: MessagesState) -> str:
    """如果最后一条 AI 消息包含 tool_calls，就进入 ToolNode。"""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


def build_analyst_tool_calling_app(
    llm_client: Any,
    tools: list[BaseTool],
    force_first_tool_name: str | None = None,
    max_tool_rounds: int = DEFAULT_ANALYST_MAX_TOOL_ROUNDS,
):
    """构建某个 Analyst 的 Tool Calling 子图。"""
    tool_schema = build_tools_schema(tools)
    actual_max_tool_rounds = max(0, int(max_tool_rounds))

    def call_agent(state: MessagesState) -> dict[str, list[AIMessage]]:
        """Analyst 节点：调用模型，让模型决定是否继续调用工具。"""
        has_tool_result = any(isinstance(message, ToolMessage) for message in state["messages"])
        tool_rounds = count_tool_call_rounds(state["messages"])
        tool_choice: str | dict[str, Any] | None = "auto"
        payload = langchain_messages_to_payload(state["messages"])

        if tool_rounds >= actual_max_tool_rounds:
            payload.append(
                {
                    "role": "user",
                    "content": (
                        "工具调用轮数已经达到程序上限。"
                        "请不要再调用工具，必须只基于以上已有工具结果，"
                        "输出一份完整、清晰、中文的分析报告。"
                    ),
                }
            )
            response = llm_client.chat(
                messages=payload,
                tools=None,
                tool_choice=None,
                temperature=0.2,
            )
            choices = response.get("choices") or []
            if not choices:
                raise ValueError("模型返回结果中没有 choices。")
            raw_message = choices[0].get("message")
            if not isinstance(raw_message, dict):
                raise ValueError("模型返回结果中没有合法 message。")
            return {"messages": [assistant_dict_to_ai_message(raw_message)]}

        # 为了保证关键数据一定进入分析，首轮可以强制调用一个核心工具。
        # 例如 Market Analyst 首轮必须调用技术面快照工具。
        if force_first_tool_name and not has_tool_result:
            tool_choice = build_force_tool_choice(force_first_tool_name)

        response = llm_client.chat(
            messages=payload,
            tools=tool_schema,
            tool_choice=tool_choice,
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
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue_tools, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    return workflow.compile(checkpointer=create_memory_checkpointer())


def run_analyst_tool_calling_report(
    agent_name: str,
    system_prompt: str,
    task_prompt: str,
    tools: list[BaseTool],
    llm_client: Any | None = None,
    force_first_tool_name: str | None = None,
    thread_id: str = "analyst-tool-calling",
    recursion_limit: int = DEFAULT_ANALYST_RECURSION_LIMIT,
    max_tool_rounds: int = DEFAULT_ANALYST_MAX_TOOL_ROUNDS,
) -> AnalystToolCallingResult:
    """运行一个带 ToolNode 的 Analyst，并返回最终报告。"""
    if not tools:
        raise ValueError(f"{agent_name} 没有配置任何工具，无法运行 Tool Calling。")

    client = llm_client or create_chat_client()
    app = build_analyst_tool_calling_app(
        llm_client=client,
        tools=tools,
        force_first_tool_name=force_first_tool_name,
        max_tool_rounds=max_tool_rounds,
    )
    readable_agent_name = localize_report_text(agent_name)
    emit_progress(f"{readable_agent_name} 正在调用大模型和工具，请稍等。")
    output = app.invoke(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=task_prompt),
            ]
        },
        config={
            **build_thread_config(thread_id),
            "recursion_limit": recursion_limit,
        },
    )
    messages = output["messages"]
    final_message = messages[-1]
    report = str(getattr(final_message, "content", "") or "").strip()
    if not report:
        raise ValueError(f"{agent_name} 没有生成有效报告。")

    tool_call_count = sum(
        len(message.tool_calls)
        for message in messages
        if isinstance(message, AIMessage)
    )
    emit_progress(f"{readable_agent_name} 已完成，工具调用 {tool_call_count} 次。")
    return AnalystToolCallingResult(
        report=report,
        messages=messages,
        tool_call_count=tool_call_count,
        tool_trace=build_tool_call_trace(messages),
    )


def count_tool_call_rounds(messages: list[BaseMessage]) -> int:
    """统计模型已经发起过几轮工具调用。

    一轮里可能同时包含多个 tool_calls。
    这里按“有 tool_calls 的 AIMessage 数量”计算轮数。
    """
    return sum(
        1
        for message in messages
        if isinstance(message, AIMessage) and bool(message.tool_calls)
    )


def convert_messages_for_debug(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """把 LangChain messages 转成现有 messages_by_agent 调试格式。"""
    return langchain_messages_to_payload(messages)


def build_tool_call_trace(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """从消息列表里提取工具调用轨迹。

    返回结构适合写进 full_state.json：
        - assistant_tool_call：模型请求调用哪个工具；
        - tool_result：ToolNode 返回了什么结果摘要。
    """
    trace: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                trace.append(
                    {
                        "message_index": index,
                        "event": "assistant_tool_call",
                        "tool_call_id": str(tool_call.get("id") or ""),
                        "tool_name": str(tool_call.get("name") or ""),
                        "args": tool_call.get("args") or {},
                    }
                )
        elif isinstance(message, ToolMessage):
            content = str(message.content or "")
            trace.append(
                {
                    "message_index": index,
                    "event": "tool_result",
                    "tool_call_id": str(getattr(message, "tool_call_id", "") or ""),
                    "tool_name": str(getattr(message, "name", "") or ""),
                    "content_length": len(content),
                    "content_preview": content[:300],
                }
            )
    return trace
