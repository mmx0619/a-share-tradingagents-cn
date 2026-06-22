"""交易记忆的大模型反思层。

原版 TradingAgents 会在知道交易结果后，
让模型根据最终决策和收益表现写一段短反思。

这个文件做同样的事情，但使用中文 Prompt：

    输入：
        最终交易决策
        个股收益
        相对 A 股基准的超额收益

    输出：
        2-4 句中文复盘

如果模型调用失败，上层会继续使用规则版反思兜底。
"""

from __future__ import annotations

from typing import Any, Protocol

from tradingagents_cn.llm.deepseek_client import extract_assistant_message


class ReflectionLLMClient(Protocol):
    """反思层需要的最小大模型客户端协议。"""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """发送聊天消息并返回原始 JSON。"""


REFLECTION_SYSTEM_PROMPT = """你是 A 股交易复盘分析员。

你会看到一次过去的最终交易决策，以及这次决策之后的收益表现。

你的任务：
1. 判断当时的方向判断是否正确。
2. 指出当时投资逻辑中哪个部分被验证，或者哪个部分失效。
3. 给出下次遇到类似情况时应该吸取的一条具体经验。

输出要求：
1. 只输出中文自然语言。
2. 必须是 2-4 句话。
3. 不要使用 Markdown。
4. 不要使用项目符号。
5. 不要重新给出买卖建议，只做复盘。
6. 文字要具体，不要写空话。"""


def reflect_on_final_decision(
    llm_client: ReflectionLLMClient,
    final_decision: str,
    raw_return: float,
    alpha_return: float,
    benchmark_name: str,
    temperature: float = 0.0,
) -> str:
    """调用大模型生成交易复盘反思。"""
    messages = [
        {
            "role": "system",
            "content": REFLECTION_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": build_reflection_user_prompt(
                final_decision=final_decision,
                raw_return=raw_return,
                alpha_return=alpha_return,
                benchmark_name=benchmark_name,
            ),
        },
    ]
    response = llm_client.chat(messages=messages, temperature=temperature)
    message = extract_assistant_message(response)
    content = str(message.get("content") or "").strip()
    if not content:
        raise ValueError("大模型没有返回反思文本。")
    return normalize_reflection_text(content)


def build_reflection_user_prompt(
    final_decision: str,
    raw_return: float,
    alpha_return: float,
    benchmark_name: str,
) -> str:
    """构造交易复盘用户 Prompt。"""
    return f"""收益结果：
个股收益：{raw_return:+.1%}
相对{benchmark_name}超额收益：{alpha_return:+.1%}

当时最终交易决策：
{final_decision}

请根据以上内容，写出 2-4 句中文复盘。"""


def normalize_reflection_text(text: str) -> str:
    """清理模型反思文本。

    这里不做复杂改写，只移除常见代码块包裹和首尾空白。
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("text"):
            cleaned = cleaned[4:].strip()
    return cleaned
