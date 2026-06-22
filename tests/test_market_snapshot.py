"""Market Agent 确定性行情快照测试。

这些测试不联网，也不调用大模型。

它们只验证 Market Agent 的“事实锚点”是否可靠：

1. 快照必须排除分析日期之后的数据，避免未来函数。
2. 快照必须使用分析日之前最近一个交易日。
3. 快照必须包含关键 OHLCV 和技术指标字段。
"""

from __future__ import annotations

import unittest

import pandas as pd

from tradingagents_cn.agents.market_agent import build_verified_market_snapshot_text


class MarketSnapshotTest(unittest.TestCase):
    """测试 Market Agent 的已校验市场数据快照。"""

    def test_snapshot_should_exclude_future_rows(self) -> None:
        """分析日期之后的数据不能进入快照。"""
        data = pd.DataFrame(
            {
                "Date": ["2026-06-17", "2026-06-18", "2026-06-19"],
                "Open": [9.8, 10.0, 99.0],
                "High": [10.2, 10.5, 99.0],
                "Low": [9.7, 9.9, 99.0],
                "Close": [10.0, 10.3, 99.0],
                "Volume": [1000, 1200, 999999],
                "rsi": [50.0, 55.0, 99.0],
                "macd": [0.1, 0.2, 9.9],
            }
        )

        snapshot = build_verified_market_snapshot_text(
            symbol="000725",
            trade_date="2026-06-18",
            indicator_data=data,
            look_back_days=5,
        )

        self.assertIn("实际使用的最近交易日：2026-06-18", snapshot)
        self.assertIn("| Close | 10.30 |", snapshot)
        self.assertIn("| rsi | 55.00 |", snapshot)
        self.assertNotIn("99.00", snapshot)
        self.assertNotIn("999999", snapshot)

    def test_snapshot_should_use_previous_trading_day_when_trade_date_missing(self) -> None:
        """如果分析日当天没有行情，应该使用之前最近交易日。"""
        data = pd.DataFrame(
            {
                "Date": ["2026-06-17", "2026-06-19"],
                "Open": [9.8, 99.0],
                "High": [10.2, 99.0],
                "Low": [9.7, 99.0],
                "Close": [10.0, 99.0],
                "Volume": [1000, 999999],
                "mfi": [60.0, 99.0],
            }
        )

        snapshot = build_verified_market_snapshot_text(
            symbol="000725",
            trade_date="2026-06-18",
            indicator_data=data,
            look_back_days=5,
        )

        self.assertIn("实际使用的最近交易日：2026-06-17", snapshot)
        self.assertIn("| Close | 10.00 |", snapshot)
        self.assertIn("| mfi | 60.00 |", snapshot)
        self.assertNotIn("99.00", snapshot)

    def test_snapshot_should_raise_when_no_rows_before_trade_date(self) -> None:
        """如果分析日前没有任何行情，应该直接报错。"""
        data = pd.DataFrame(
            {
                "Date": ["2026-06-19"],
                "Open": [10.0],
                "High": [10.0],
                "Low": [10.0],
                "Close": [10.0],
                "Volume": [1000],
            }
        )

        with self.assertRaises(ValueError):
            build_verified_market_snapshot_text(
                symbol="000725",
                trade_date="2026-06-18",
                indicator_data=data,
            )


if __name__ == "__main__":
    unittest.main()
