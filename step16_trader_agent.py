"""第 16 步：交易员 Agent。

第 15 步已经完成风控 Agent：

综合投研汇总报告
  ↓
风控 Agent
  ↓
风险等级、禁止动作、观察条件、仓位约束

当前文件做第 16 件事：

风控报告
  ↓
交易员 Prompt
  ↓
大模型
  ↓
条件式交易预案

交易员 Agent 的定位：
它不是无条件下单。
它必须服从风控 Agent 的边界。

如果风控报告明确禁止抄底、禁止追涨、禁止重仓，
交易员 Agent 就不能绕过这些限制。

所以这个文件输出的是“交易预案”，不是实际交易指令。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import step08_llm_client as llm_mod
import step15_risk_control_agent as risk_agent


@dataclass
class TraderPlanResult:
    """交易员 Agent 的返回结果。

    字段说明：
    - symbol：股票代码。
    - risk_report：第 15 步风控 Agent 报告。
    - prompt：发送给交易员 Agent 的完整 Prompt。
    - provider：模型平台，比如 deepseek。
    - model：具体模型名称。
    - trader_plan：交易员 Agent 输出的条件式交易预案。
    """

    symbol: str
    risk_report: str
    prompt: str
    provider: str
    model: str | None
    trader_plan: str


def build_trader_prompt(symbol: str, risk_report: str) -> str:
    """生成交易员 Agent 的 Prompt。

    交易员 Agent 只接收风控报告。
    这样设计是为了强调：
    交易员不能绕过风控自己找理由交易。

    后续如果要更像 TradingAgents，可以把技术面、新闻面、风控报告都给它；
    但在当前学习阶段，先让它严格服从风控边界。
    """
    return f"""你是一名 A 股多智能体投研系统里的交易员 Agent。

你现在收到的是风控 Agent 的报告。

你的任务不是直接下单，
而是在风控边界内制定“条件式交易预案”。

请遵守以下要求：

1. 必须服从风控报告中的禁止动作和仓位约束。
2. 如果风控报告禁止参与，交易预案必须是观望或拒绝交易。
3. 不要输出确定性的买入、卖出、持有指令。
4. 只允许输出条件式方案，比如“只有满足 A、B、C 后，才允许重新评估”。
5. 如果风险等级是高或极高，默认立场必须偏防守。
6. 仓位只能写成上限和约束，不允许写成确定下单比例。

输出格式：

## 交易员结论
用 2-4 句话说明当前是否允许讨论交易方案。

## 当前动作
从以下选项中选择一个：
- 观望
- 拒绝交易
- 等待条件触发
- 仅允许极低仓位观察

## 触发条件
列出只有满足哪些条件，才允许进入下一轮交易讨论。

## 无效条件
列出哪些情况一旦出现，交易预案立即失效。

## 仓位边界
严格复述或收紧风控 Agent 给出的仓位约束。

## 给后续执行系统的说明
说明后续如果要真正执行，还需要哪些实时数据确认。

股票代码：
{symbol}

风控 Agent 报告：
{risk_report}
"""


def run_trader_agent(
    symbol: str,
    risk_report: str,
    provider: str = "deepseek",
    model: str | None = None,
    temperature: float = 0.2,
) -> TraderPlanResult:
    """运行交易员 Agent。"""
    prompt = build_trader_prompt(
        symbol=symbol,
        risk_report=risk_report,
    )
    client = llm_mod.create_llm_client(
        provider=provider,
        model=model,
        temperature=temperature,
    )
    response = llm_mod.call_llm(client, prompt)

    return TraderPlanResult(
        symbol=symbol,
        risk_report=risk_report,
        prompt=prompt,
        provider=response.provider,
        model=response.model,
        trader_plan=response.text,
    )


def run_full_trader_pipeline(
    symbol: str,
    start_date: str,
    end_date: str,
    news_max_items: int = 5,
    provider: str = "deepseek",
    model: str | None = None,
) -> TraderPlanResult:
    """运行完整交易员链路。

    当前完整链路：
    1. 第 15 步先生成风控报告。
    2. 第 16 步把风控报告交给交易员 Agent。
    3. 交易员 Agent 输出条件式交易预案。
    """
    risk_result = risk_agent.run_full_risk_control_pipeline(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        news_max_items=news_max_items,
        provider=provider,
        model=model,
    )

    return run_trader_agent(
        symbol=risk_result.symbol,
        risk_report=risk_result.risk_report,
        provider=provider,
        model=model,
    )


def render_trader_plan_result(result: TraderPlanResult) -> str:
    """把交易员 Agent 的结果渲染成方便阅读的文本。"""
    return f"""股票代码：{result.symbol}
模型来源：{result.provider}
模型名称：{result.model}

======== 风控 Agent 报告 ========
{result.risk_report}

======== 交易员 Agent 预案 ========
{result.trader_plan}
"""


def demo_trader_pipeline() -> None:
    """演示完整交易员链路。

    默认继续使用 002361。
    这个例子可以测试交易员 Agent 是否会服从风控边界。
    """
    symbol = os.environ.get("STOCK_SYMBOL", "002361")
    start_date = os.environ.get("START_DATE", "2026-01-01")
    end_date = os.environ.get("END_DATE", "2026-06-15")
    news_max_items = int(os.environ.get("NEWS_MAX_ITEMS", "5"))
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model = os.environ.get("LLM_MODEL") or None

    result = run_full_trader_pipeline(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        news_max_items=news_max_items,
        provider=provider,
        model=model,
    )
    print(render_trader_plan_result(result))


if __name__ == "__main__":
    demo_trader_pipeline()
