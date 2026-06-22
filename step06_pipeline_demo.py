"""第 06 步：把 01 到 05 串成第一条最小流水线。

当前文件的目标：
不用真实 AKShare，不用真实大模型，只用一份本地演示行情数据，
把前面几个模块串起来，跑通一条完整流程。

当前最小流水线：

演示行情表
  ↓
step02_technical_indicators.py 计算技术指标
  ↓
step03_market_snapshot.py 生成市场快照
  ↓
step04_market_report_prompt.py 生成市场分析师 Prompt
  ↓
step05_market_report_mock_llm.py 模拟大模型返回市场分析报告

为什么不用真实 AKShare：
这一步的重点不是数据源，而是理解“模块如何串联”。
真实 AKShare 获取数据已经在第 01 步单独负责。

为什么不用真实大模型：
这一步的重点不是模型能力，而是理解“Prompt 如何进入 Agent，报告如何返回”。
真实大模型接入会放到后面的步骤。
"""

from __future__ import annotations

import pandas as pd

import step02_technical_indicators as technical
import step03_market_snapshot as snapshot_mod
import step04_market_report_prompt as prompt_mod
import step05_market_report_mock_llm as mock_llm_mod


def build_demo_price_data() -> pd.DataFrame:
    """构造一份本地演示行情数据。

    这份数据模拟一只股票最近 10 个交易日的日线行情。

    注意：
    这里的数据是手写的演示数据，不代表真实股票走势。
    它只是为了让流水线能离线跑通。
    """
    return pd.DataFrame(
        [
            {"Date": "2024-01-02", "Open": 10.0, "High": 10.3, "Low": 9.8, "Close": 10.1, "Volume": 1000},
            {"Date": "2024-01-03", "Open": 10.1, "High": 10.5, "Low": 10.0, "Close": 10.4, "Volume": 1100},
            {"Date": "2024-01-04", "Open": 10.4, "High": 10.6, "Low": 10.1, "Close": 10.2, "Volume": 1050},
            {"Date": "2024-01-05", "Open": 10.2, "High": 10.7, "Low": 10.2, "Close": 10.6, "Volume": 1300},
            {"Date": "2024-01-08", "Open": 10.6, "High": 10.9, "Low": 10.4, "Close": 10.8, "Volume": 1500},
            {"Date": "2024-01-09", "Open": 10.8, "High": 11.0, "Low": 10.6, "Close": 10.7, "Volume": 1250},
            {"Date": "2024-01-10", "Open": 10.7, "High": 11.2, "Low": 10.7, "Close": 11.1, "Volume": 1800},
            {"Date": "2024-01-11", "Open": 11.1, "High": 11.5, "Low": 11.0, "Close": 11.4, "Volume": 2100},
            {"Date": "2024-01-12", "Open": 11.4, "High": 11.6, "Low": 11.1, "Close": 11.2, "Volume": 1700},
            {"Date": "2024-01-15", "Open": 11.2, "High": 11.9, "Low": 11.2, "Close": 11.8, "Volume": 2600},
        ]
    )


def run_demo_pipeline(symbol: str = "600519") -> str:
    """运行第一条最小流水线，并返回最终市场分析报告文本。"""
    # 第 1 步：准备演示行情数据。
    price_data = build_demo_price_data()

    # 第 2 步：计算技术指标。
    indicator_data = technical.build_basic_technical_indicators(price_data)

    # 第 3 步：把最新交易日整理成市场快照。
    snapshot = snapshot_mod.build_market_snapshot(symbol, indicator_data)
    snapshot_text = snapshot_mod.render_market_snapshot_text(snapshot)

    # 第 4 步：把市场快照转换成市场分析师 Prompt。
    prompt = prompt_mod.build_market_prompt_from_text(symbol, snapshot_text)

    # 第 5 步：用 mock LLM 模拟市场分析师输出报告。
    report = mock_llm_mod.run_market_report_agent(symbol, prompt)

    return mock_llm_mod.render_market_report(report)


if __name__ == "__main__":
    final_report = run_demo_pipeline("600519")
    print(final_report)
