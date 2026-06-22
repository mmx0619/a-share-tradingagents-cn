"""A 股市场概览数据层测试。

这些测试不联网。

它们只验证市场概览原材料的整理规则：
1. 指数快照字段标准化；
2. 涨跌家数统计；
3. 行业板块按涨跌幅排序；
4. 文本渲染包含关键部分。
"""

from __future__ import annotations

import unittest

import pandas as pd

from tradingagents_cn.dataflows.market_overview import (
    MarketOverview,
    calculate_market_breadth,
    normalize_index_snapshot,
    normalize_sector_snapshot,
    render_market_overview_text,
)


class MarketOverviewTest(unittest.TestCase):
    """测试市场概览数据整理。"""

    def test_calculate_market_breadth(self) -> None:
        """涨跌家数应该按涨跌幅正负统计。"""
        data = pd.DataFrame({"涨跌幅": [1.2, -0.5, 0, 3.0, -2.1]})

        breadth = calculate_market_breadth(data)

        self.assertEqual(breadth.total_count, 5)
        self.assertEqual(breadth.up_count, 2)
        self.assertEqual(breadth.down_count, 2)
        self.assertEqual(breadth.flat_count, 1)
        self.assertAlmostEqual(breadth.up_ratio, 0.4)

    def test_normalize_index_snapshot(self) -> None:
        """指数快照应该统一成 Name / Latest / ChangePct / Amount。"""
        data = pd.DataFrame(
            {
                "名称": ["上证指数", "深证成指", "其他指数"],
                "最新价": [3000.1, 9800.2, 123.4],
                "涨跌幅": [0.5, -0.2, 1.0],
                "成交额": [100000, 200000, 3000],
            }
        )

        normalized = normalize_index_snapshot(data)

        self.assertIn("Name", normalized.columns)
        self.assertIn("ChangePct", normalized.columns)
        self.assertIn("上证指数", normalized["Name"].tolist())
        self.assertIn("深证成指", normalized["Name"].tolist())

    def test_normalize_sector_snapshot_should_sort_by_change_pct(self) -> None:
        """行业板块应该按涨跌幅从高到低排序。"""
        data = pd.DataFrame(
            {
                "板块名称": ["银行", "半导体", "煤炭"],
                "涨跌幅": [-0.5, 2.3, 1.1],
                "成交额": [100, 300, 200],
                "领涨股票": ["A", "B", "C"],
            }
        )

        normalized = normalize_sector_snapshot(data, top_n=2)

        self.assertEqual(normalized.iloc[0]["Name"], "半导体")
        self.assertEqual(normalized.iloc[1]["Name"], "煤炭")
        self.assertEqual(len(normalized), 2)

    def test_render_market_overview_text(self) -> None:
        """市场概览文本必须包含指数、涨跌家数和行业板块。"""
        overview = MarketOverview(
            index_snapshot=pd.DataFrame(
                {"Name": ["上证指数"], "Latest": [3000], "ChangePct": [0.5], "Amount": [1000]}
            ),
            market_breadth=calculate_market_breadth(
                pd.DataFrame({"涨跌幅": [1.0, -1.0, 0.0]})
            ),
            sector_snapshot=pd.DataFrame(
                {"Name": ["半导体"], "ChangePct": [2.0], "Amount": [100], "LeadingStock": ["B"]}
            ),
        )

        text = render_market_overview_text(overview)

        self.assertIn("A 股市场概览原材料", text)
        self.assertIn("主要指数", text)
        self.assertIn("市场涨跌家数", text)
        self.assertIn("行业板块强弱", text)
        self.assertIn("上涨家数：1", text)


if __name__ == "__main__":
    unittest.main()
