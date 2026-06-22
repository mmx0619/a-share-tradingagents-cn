"""第 07 步：把真实 AKShare 行情接入最小流水线。

第 06 步使用的是本地演示行情数据。
当前文件改成使用第 01 步的 AKShare 函数获取真实 A 股日线数据。

当前真实数据流水线：

AKShare 获取真实日线行情
  ↓
step02_technical_indicators.py 计算技术指标
  ↓
step03_market_snapshot.py 生成市场快照
  ↓
step04_market_report_prompt.py 生成市场分析师 Prompt
  ↓
step05_market_report_mock_llm.py 模拟大模型返回市场分析报告

注意：
1. 这个文件会调用 AKShare，需要联网，也需要安装 akshare。
2. 这个文件仍然不调用真实大模型 API。
3. 当前输出的市场分析报告仍然来自 mock LLM，不是 OpenAI/DeepSeek/Qwen。
"""

from __future__ import annotations

import step01_akshare_cn as akshare_mod
import step02_technical_indicators as technical
import step03_market_snapshot as snapshot_mod
import step04_market_report_prompt as prompt_mod
import step05_market_report_mock_llm as mock_llm_mod


def run_real_akshare_pipeline(
    symbol: str,
    start_date: str,
    end_date: str,
) -> str:
    """运行真实 AKShare 行情版最小流水线。

    参数：
    - symbol：A 股代码，比如 600519、000001、300750。
    - start_date：开始日期，格式 YYYY-MM-DD。
    - end_date：结束日期，格式 YYYY-MM-DD。

    返回：
    - mock 市场分析报告文本。
    """
    # 第 1 步：通过 AKShare 获取真实 A 股日线行情。
    price_data = akshare_mod.get_a_share_daily_history(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )

    # 第 2 步：计算基础技术指标。
    indicator_data = technical.build_basic_technical_indicators(price_data)

    # 第 3 步：整理最新交易日市场快照。
    normalized_symbol = akshare_mod.normalize_cn_symbol(symbol)
    snapshot = snapshot_mod.build_market_snapshot(normalized_symbol, indicator_data)
    snapshot_text = snapshot_mod.render_market_snapshot_text(snapshot)

    # 第 4 步：生成市场分析师 Prompt。
    prompt = prompt_mod.build_market_prompt_from_text(normalized_symbol, snapshot_text)

    # 第 5 步：用 mock LLM 模拟市场分析师输出报告。
    report = mock_llm_mod.run_market_report_agent(normalized_symbol, prompt)
    return mock_llm_mod.render_market_report(report)


if __name__ == "__main__":
    # 这里默认演示贵州茅台 600519。
    # 如果要换股票，可以改 symbol、start_date、end_date。
    final_report = run_real_akshare_pipeline(
        symbol="002361",
        start_date="2026-06-12",
        end_date="2026-06-12",
    )
    print(final_report)
