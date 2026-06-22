"""第 22 步：实时行情感知的风控 Agent。

第 15 步的风控 Agent 读取的是普通综合投研报告。
第 21 步已经生成了“实时行情感知综合报告”。

当前文件做第 22 件事：

实时行情感知综合报告
  ↓
实时风控 Prompt
  ↓
大模型
  ↓
实时风控报告

这个 Agent 要回答的问题：

- 实时行情是否足以改变风险等级？
- 如果只是小幅反弹，是否仍然禁止交易？
- 如果风险等级维持高位，交易员 Agent 的边界应该怎么写？

注意：
这一步仍然不直接给交易指令。
它只负责更新风险边界。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import step08_llm_client as llm_mod
import step17_agent_state as agent_state
import step20_realtime_stateful_workflow as realtime_workflow
import step21_realtime_summary_agent as realtime_summary


@dataclass
class RealtimeRiskControlResult:
    """实时风控 Agent 的返回结果。

    字段说明：
    - symbol：股票代码。
    - realtime_summary_text：第 21 步实时综合报告。
    - prompt：发送给模型的完整 Prompt。
    - provider：模型平台。
    - model：具体模型名称。
    - risk_report：实时风控报告。
    """

    symbol: str
    realtime_summary_text: str
    prompt: str
    provider: str
    model: str | None
    risk_report: str


def build_realtime_risk_prompt(symbol: str, realtime_summary_text: str) -> str:
    """生成实时风控 Prompt。

    这个 Prompt 的核心：
    风控 Agent 不再只看历史综合结论，
    而是看“实时综合 Agent”已经处理过的结果。
    """
    return f"""你是一名 A 股多智能体投研系统里的实时风控 Agent。

你现在收到的是“实时行情感知综合投研报告”。

你的任务不是寻找交易机会，
而是根据实时综合报告更新风险边界。

请遵守以下要求：

1. 只基于我提供的实时综合报告分析，不要编造新行情。
2. 不要输出确定性的买入或卖出指令。
3. 如果实时行情只是小幅反弹，但报告认为没有改变弱势结构，必须维持高风险判断。
4. 如果报告建议维持或上调风险等级，必须明确禁止动作。
5. 如果报告建议下调风险等级，也必须说明下调前需要哪些确认条件。
6. 必须给出交易员 Agent 下一步能做什么、不能做什么。

输出格式：

## 实时风控结论
用 2-4 句话说明实时行情是否改变风险边界。

## 风险等级调整
从 维持 / 上调 / 下调 中选择一个，并说明当前风险等级。

## 禁止动作
列出当前不允许交易员 Agent 做的事情。

## 允许观察项
列出交易员 Agent 可以继续观察但不能直接交易的内容。

## 风控触发条件
列出哪些条件出现后，风险等级需要重新评估。

## 给交易员 Agent 的实时边界
明确交易员 Agent 当前只能做什么。

股票代码：
{symbol}

实时行情感知综合投研报告：
{realtime_summary_text}
"""


def run_realtime_risk_control_agent(
    symbol: str,
    realtime_summary_text: str,
    provider: str = "deepseek",
    model: str | None = None,
    temperature: float = 0.2,
) -> RealtimeRiskControlResult:
    """运行实时风控 Agent。"""
    prompt = build_realtime_risk_prompt(
        symbol=symbol,
        realtime_summary_text=realtime_summary_text,
    )
    client = llm_mod.create_llm_client(
        provider=provider,
        model=model,
        temperature=temperature,
    )
    response = llm_mod.call_llm(client, prompt)

    return RealtimeRiskControlResult(
        symbol=symbol,
        realtime_summary_text=realtime_summary_text,
        prompt=prompt,
        provider=response.provider,
        model=response.model,
        risk_report=response.text,
    )


def run_realtime_risk_from_state(
    state: agent_state.TradingAgentState,
    provider: str = "deepseek",
    model: str | None = None,
) -> RealtimeRiskControlResult:
    """从状态中先运行实时综合 Agent，再运行实时风控 Agent。"""
    summary_result = realtime_summary.run_realtime_summary_from_state(
        state=state,
        provider=provider,
        model=model,
    )
    return run_realtime_risk_control_agent(
        symbol=state.symbol,
        realtime_summary_text=summary_result.summary_text,
        provider=provider,
        model=model,
    )


def render_realtime_risk_result(result: RealtimeRiskControlResult) -> str:
    """把实时风控结果渲染成方便阅读的文本。"""
    return f"""股票代码：{result.symbol}
模型来源：{result.provider}
模型名称：{result.model}

======== 实时综合报告 ========
{result.realtime_summary_text}

======== 实时风控报告 ========
{result.risk_report}
"""


def demo_realtime_risk_control_agent() -> None:
    """演示实时风控 Agent。

    为了有完整上下文，这里先运行第 20 步得到 state，
    再运行实时综合 Agent 和实时风控 Agent。
    """
    symbol = os.environ.get("STOCK_SYMBOL", "002361")
    start_date = os.environ.get("START_DATE", "2026-01-01")
    end_date = os.environ.get("END_DATE", "2026-06-15")
    news_max_items = int(os.environ.get("NEWS_MAX_ITEMS", "5"))
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model = os.environ.get("LLM_MODEL") or None

    state = realtime_workflow.run_realtime_stateful_workflow(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        news_max_items=news_max_items,
        provider=provider,
        model=model,
    )
    result = run_realtime_risk_from_state(
        state=state,
        provider=provider,
        model=model,
    )
    print(render_realtime_risk_result(result))


if __name__ == "__main__":
    demo_realtime_risk_control_agent()
