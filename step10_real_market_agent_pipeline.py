"""第 10 步：把第 1 步到第 9 步串成真实市场分析链路。

当前文件回答一个核心问题：

前面写了 step01 到 step09，那么真实运行时到底怎么串起来？

真实主链路是：

step01_akshare_cn.py
  获取真实 A 股行情
  ↓
step02_technical_indicators.py
  计算技术指标
  ↓
step03_market_snapshot.py
  生成市场快照
  ↓
step09_market_agent.py
  运行市场分析师 Agent
  ↓
step04_market_report_prompt.py
  在 Agent 内部生成 Prompt
  ↓
step08_llm_client.py
  在 Agent 内部调用真实大模型
  ↓
输出真实市场分析报告

为什么没有直接调用 step05、step06、step07：

- step05_market_report_mock_llm.py 是 mock 大模型，真实链路不用它。
- step06_pipeline_demo.py 是本地演示流水线，真实链路不用它。
- step07_real_akshare_pipeline.py 是早期“真实行情 + mock 报告”的练习版，
  现在已经被当前第 10 步替代。

所以第 10 步不是简单把所有文件从 01 调到 09 全部调用一遍，
而是把“真实主线”串起来。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import step01_akshare_cn as akshare_mod
import step02_technical_indicators as technical
import step03_market_snapshot as snapshot_mod
import step09_market_agent as market_agent


@dataclass
class RealMarketPipelineResult:
    """真实市场分析流水线的返回结果。

    字段说明：
    - symbol：标准化后的股票代码。
    - start_date：行情开始日期。
    - end_date：行情结束日期。
    - market_snapshot_text：第 03 步生成的市场快照文本。
    - prompt：第 09 步内部生成并发送给模型的完整提示词。
    - provider：大模型平台，比如 deepseek。
    - model：具体模型名称。
    - report_text：市场分析师 Agent 生成的报告。

    为什么保存 market_snapshot_text：
    因为报告是根据快照生成的。
    后面如果报告看起来奇怪，可以先回头检查快照是否正确。
    """

    symbol: str
    start_date: str
    end_date: str
    market_snapshot_text: str
    prompt: str
    provider: str
    model: str | None
    report_text: str


def run_real_market_agent_pipeline(
    symbol: str,
    start_date: str,
    end_date: str,
    provider: str = "deepseek",
    model: str | None = None,
) -> RealMarketPipelineResult:
    """运行真实市场分析流水线。

    参数说明：
    - symbol：A 股代码，比如 600519、000001、300750。
    - start_date：开始日期，格式 YYYY-MM-DD。
    - end_date：结束日期，格式 YYYY-MM-DD。
    - provider：大模型平台，默认 deepseek。
      也可以传 mock、openai、gemini、kimi。
    - model：具体模型名，不传则使用第 08 步默认模型。

    返回：
    - RealMarketPipelineResult。

    当前完整流程：
    1. step01：获取真实 A 股日线行情。
    2. step02：计算技术指标。
    3. step03：生成市场快照。
    4. step09：调用市场分析师 Agent。
    5. step09 内部调用 step04：生成 Prompt。
    6. step09 内部调用 step08：调用真实大模型。
    7. 返回真实大模型报告。
    """
    normalized_symbol = akshare_mod.normalize_cn_symbol(symbol)

    # 第 1 步：获取真实 A 股历史行情。
    price_data = akshare_mod.get_a_share_daily_history(
        symbol=normalized_symbol,
        start_date=start_date,
        end_date=end_date,
    )

    # 第 2 步：计算 MA、涨跌幅、量能、日内位置等基础技术指标。
    indicator_data = technical.build_basic_technical_indicators(price_data)

    # 第 3 步：把最新一日的指标整理成市场快照。
    snapshot = snapshot_mod.build_market_snapshot(normalized_symbol, indicator_data)
    market_snapshot_text = snapshot_mod.render_market_snapshot_text(snapshot)

    # 第 4 步：把市场快照交给市场分析师 Agent。
    agent_result = market_agent.run_market_agent(
        symbol=normalized_symbol,
        market_snapshot_text=market_snapshot_text,
        provider=provider,
        model=model,
    )

    return RealMarketPipelineResult(
        symbol=normalized_symbol,
        start_date=start_date,
        end_date=end_date,
        market_snapshot_text=market_snapshot_text,
        prompt=agent_result.prompt,
        provider=agent_result.provider,
        model=agent_result.model,
        report_text=agent_result.report_text,
    )


def render_real_market_pipeline_result(result: RealMarketPipelineResult) -> str:
    """把真实市场分析流水线结果渲染成方便阅读的文本。"""
    return f"""股票代码：{result.symbol}
行情区间：{result.start_date} 至 {result.end_date}
模型来源：{result.provider}
模型名称：{result.model}

======== 市场快照 ========
{result.market_snapshot_text}

======== 发送给大模型的 Prompt ========
{result.prompt}

======== 市场分析师报告 ========
{result.report_text}
"""


def demo_real_market_pipeline() -> None:
    """演示真实行情 + 真实大模型分析。

    默认参数：
    - 股票：600519
    - 模型：deepseek

    如果想临时切换，可以设置环境变量：
    - STOCK_SYMBOL=000001
    - START_DATE=2026-01-01
    - END_DATE=2026-06-12
    - LLM_PROVIDER=mock / deepseek / openai / gemini / kimi
    - LLM_MODEL=具体模型名
    """
    symbol = os.environ.get("STOCK_SYMBOL", "600519")
    start_date = os.environ.get("START_DATE", "2026-01-01")
    end_date = os.environ.get("END_DATE", "2026-06-12")
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model = os.environ.get("LLM_MODEL") or None

    result = run_real_market_agent_pipeline(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        provider=provider,
        model=model,
    )
    print(render_real_market_pipeline_result(result))


if __name__ == "__main__":
    demo_real_market_pipeline()
