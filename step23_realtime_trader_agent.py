"""第 23 步：实时交易员 Agent。

第 16 步的交易员 Agent 读取普通风控报告。
第 22 步已经生成了实时风控报告。

当前文件做第 23 件事：

实时风控报告
  ↓
实时交易员 Prompt
  ↓
大模型
  ↓
实时交易预案

这个 Agent 要回答的问题：

- 当前实时状态下，交易员是否允许行动？
- 如果风控禁止开仓，交易员是否严格服从？
- 如果只允许观察，具体观察哪些实时条件？

注意：
这里依然不是真实下单系统。
它只是生成“实时条件式交易预案”。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import step08_llm_client as llm_mod
import step20_realtime_stateful_workflow as realtime_workflow
import step22_realtime_risk_control_agent as realtime_risk


@dataclass
class RealtimeTraderResult:
    """实时交易员 Agent 的返回结果。

    字段说明：
    - symbol：股票代码。
    - realtime_risk_report：第 22 步实时风控报告。
    - prompt：发送给模型的完整 Prompt。
    - provider：模型平台。
    - model：具体模型名称。
    - trader_plan：实时交易员预案。
    """

    symbol: str
    realtime_risk_report: str
    prompt: str
    provider: str
    model: str | None
    trader_plan: str


def build_realtime_trader_prompt(symbol: str, realtime_risk_report: str) -> str:
    """生成实时交易员 Prompt。

    交易员必须服从实时风控报告。
    如果实时风控报告禁止主动开仓，
    交易员不能自己找理由突破限制。
    """
    return f"""你是一名 A 股多智能体投研系统里的实时交易员 Agent。

你现在收到的是实时风控 Agent 的报告。

你的任务不是直接下单，
而是在实时风控边界内制定当前时点的条件式交易预案。

请遵守以下要求：

1. 必须服从实时风控报告中的禁止动作和实时边界。
2. 如果实时风控报告禁止主动开仓，当前动作必须是观望或拒绝交易。
3. 不要输出确定性的买入、卖出、持有指令。
4. 只允许输出实时条件式方案。
5. 如果风险等级仍为高或极高，默认立场必须防守。
6. 如果允许观察，必须说明观察哪些实时数据。

输出格式：

## 实时交易员结论
用 2-4 句话说明当前是否允许行动。

## 当前动作
从以下选项中选择一个：
- 观望
- 拒绝交易
- 等待实时条件触发
- 仅允许处理已有风险敞口

## 实时观察清单
列出当前可以继续观察的实时数据。

## 触发条件
列出只有满足哪些实时条件，才允许重新讨论交易方案。

## 失效条件
列出哪些情况出现后，预案立即失效。

## 仓位边界
严格服从实时风控报告，不允许自行放宽。

## 给执行系统的说明
说明如果未来进入执行阶段，还需要哪些实时确认。

股票代码：
{symbol}

实时风控 Agent 报告：
{realtime_risk_report}
"""


def run_realtime_trader_agent(
    symbol: str,
    realtime_risk_report: str,
    provider: str = "deepseek",
    model: str | None = None,
    temperature: float = 0.2,
) -> RealtimeTraderResult:
    """运行实时交易员 Agent。"""
    prompt = build_realtime_trader_prompt(
        symbol=symbol,
        realtime_risk_report=realtime_risk_report,
    )
    client = llm_mod.create_llm_client(
        provider=provider,
        model=model,
        temperature=temperature,
    )
    response = llm_mod.call_llm(client, prompt)

    return RealtimeTraderResult(
        symbol=symbol,
        realtime_risk_report=realtime_risk_report,
        prompt=prompt,
        provider=response.provider,
        model=response.model,
        trader_plan=response.text,
    )


def run_full_realtime_trader_pipeline(
    symbol: str,
    start_date: str,
    end_date: str,
    news_max_items: int = 5,
    provider: str = "deepseek",
    model: str | None = None,
) -> RealtimeTraderResult:
    """运行完整实时交易员链路。

    当前完整链路：
    1. 第 20 步生成包含实时行情的 state。
    2. 第 21 步生成实时综合报告。
    3. 第 22 步生成实时风控报告。
    4. 第 23 步生成实时交易预案。
    """
    state = realtime_workflow.run_realtime_stateful_workflow(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        news_max_items=news_max_items,
        provider=provider,
        model=model,
    )
    risk_result = realtime_risk.run_realtime_risk_from_state(
        state=state,
        provider=provider,
        model=model,
    )
    return run_realtime_trader_agent(
        symbol=symbol,
        realtime_risk_report=risk_result.risk_report,
        provider=provider,
        model=model,
    )


def render_realtime_trader_result(result: RealtimeTraderResult) -> str:
    """把实时交易员结果渲染成方便阅读的文本。"""
    return f"""股票代码：{result.symbol}
模型来源：{result.provider}
模型名称：{result.model}

======== 实时风控报告 ========
{result.realtime_risk_report}

======== 实时交易员预案 ========
{result.trader_plan}
"""


def demo_realtime_trader_agent() -> None:
    """演示完整实时交易员链路。"""
    symbol = os.environ.get("STOCK_SYMBOL", "002361")
    start_date = os.environ.get("START_DATE", "2026-01-01")
    end_date = os.environ.get("END_DATE", "2026-06-15")
    news_max_items = int(os.environ.get("NEWS_MAX_ITEMS", "5"))
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model = os.environ.get("LLM_MODEL") or None

    result = run_full_realtime_trader_pipeline(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        news_max_items=news_max_items,
        provider=provider,
        model=model,
    )
    print(render_realtime_trader_result(result))


if __name__ == "__main__":
    demo_realtime_trader_agent()
