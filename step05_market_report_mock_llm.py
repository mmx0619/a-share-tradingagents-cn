"""第 05 步：用假的大模型模拟市场分析师返回报告。

当前文件仍然不调用真实大模型 API。

为什么要先写 mock LLM：
1. 真实大模型需要 API Key，会增加调试成本。
2. 我们现在重点是理解数据流，不是马上花钱调用模型。
3. 先用假的模型返回固定格式报告，可以看清楚：
   市场快照 -> Prompt -> 模型返回文本 -> 下一步程序使用报告。

这里的 mock LLM 可以理解为：
一个临时替身。
未来接入 OpenAI、DeepSeek、Qwen 时，会把这个替身换成真实模型调用。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MarketReport:
    """市场分析报告的数据结构。

    这里先用非常简单的字段：
    - symbol：股票代码。
    - report_text：市场分析师输出的完整报告文本。
    - source：报告来源。当前是 mock_llm，后续可以变成 openai/deepseek/qwen。
    """

    symbol: str
    report_text: str
    source: str = "mock_llm"


def mock_market_llm(prompt: str) -> str:
    """假的市场分析师大模型。

    真实大模型会阅读完整 Prompt，然后生成一份分析报告。
    这里为了学习流程，只做一个非常简单的模拟：
    - 如果 Prompt 里出现“高于 MA5”，就认为短线偏强。
    - 如果 Prompt 里出现“低于 MA5”，就认为短线偏弱。
    - 如果 Prompt 里出现“成交量明显放大”，就提示量能增强。
    - 如果 Prompt 里出现“尾盘表现偏强”，就提示日内承接较好。

    注意：
    这不是交易策略，只是模拟大模型读 Prompt 后会产出报告。
    """
    short_trend = "短线趋势暂时不明确"
    if "高于 MA5" in prompt:
        short_trend = "短线价格站上 MA5，技术面偏强"
    elif "低于 MA5" in prompt:
        short_trend = "短线价格跌破 MA5，技术面偏弱"

    volume_view = "量能变化不明显"
    if "成交量明显放大" in prompt or "成交量温和放大" in prompt:
        volume_view = "成交量出现放大，说明交易活跃度提高"
    elif "成交量明显缩小" in prompt:
        volume_view = "成交量明显缩小，说明资金参与度下降"

    intraday_view = "日内收盘位置中性"
    if "尾盘表现偏强" in prompt:
        intraday_view = "收盘接近日内高位，尾盘承接较好"
    elif "尾盘表现偏弱" in prompt:
        intraday_view = "收盘接近日内低位，尾盘抛压较明显"

    return f"""## 技术面结论
{short_trend}。{volume_view}。{intraday_view}。

## 关键证据
- 市场快照中包含收盘价与均线的相对位置。
- 市场快照中包含成交量相对 5 日均量的变化。
- 市场快照中包含收盘价在当日 K 线中的位置。

## 风险与分歧
- 当前报告只基于技术面快照，没有结合新闻、公告、基本面和情绪面。
- 如果 MA20 或更长周期数据不足，中期趋势判断需要谨慎。
- 技术面信号不能单独作为最终交易依据。

## 给后续 Agent 的提示
- 新闻 Agent 需要检查最近是否有公告、政策或行业催化。
- 情绪 Agent 需要检查股吧、雪球等社区是否存在一致预期或过热风险。
- 风控 Agent 需要结合波动率、支撑位和仓位管理进一步判断。"""


def run_market_report_agent(symbol: str, prompt: str) -> MarketReport:
    """运行市场分析师 Agent 的最小模拟版本。

    当前版本做的事情：
    1. 接收第 04 步生成的 Prompt。
    2. 把 Prompt 交给 mock_market_llm。
    3. 把返回文本包装成 MarketReport。

    未来真实版本会把第 2 步替换成：
    - OpenAI API
    - DeepSeek API
    - Qwen API
    - 本地 Ollama 模型
    """
    report_text = mock_market_llm(prompt)
    return MarketReport(
        symbol=symbol,
        report_text=report_text,
    )


def render_market_report(report: MarketReport) -> str:
    """把 MarketReport 渲染成便于阅读的文本。"""
    return f"""股票代码：{report.symbol}
报告来源：{report.source}

{report.report_text}
"""


if __name__ == "__main__":
    # 这里放一个简化 Prompt，用于演示 mock LLM 的效果。
    demo_prompt = """股票代码：600519
市场快照：
- 收盘价高于 MA5，短期价格相对强于该均线。
- 成交量温和放大，约为 5 日均量的 1.20 倍。
- 收盘接近日内高位，说明尾盘表现偏强。"""

    report = run_market_report_agent("600519", demo_prompt)
    print(render_market_report(report))
