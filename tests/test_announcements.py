import unittest

import pandas as pd

from tradingagents_cn.dataflows.announcements import (
    announcement_frame_to_items,
    calculate_announcement_start_date,
    normalize_announcement_frame,
    render_stock_announcements_text,
    to_akshare_date,
)
from tradingagents_cn.tools.announcement_tools import get_announcement_tools
from tradingagents_cn.tools.registry import get_tool_map


class AnnouncementDataflowTest(unittest.TestCase):
    def test_normalize_cninfo_announcement_frame(self):
        """巨潮公告字段应整理成统一字段。"""
        raw = pd.DataFrame(
            {
                "代码": ["000725"],
                "简称": ["京东方A"],
                "公告标题": ["京东方A：2025 年年度报告"],
                "公告时间": ["2026-04-20 18:00:00"],
                "公告链接": ["http://www.cninfo.com.cn/test"],
            }
        )

        normalized = normalize_announcement_frame(raw, symbol="000725", source="巨潮资讯")

        self.assertEqual("000725", normalized.iloc[0]["Symbol"])
        self.assertEqual("京东方A", normalized.iloc[0]["StockName"])
        self.assertEqual("京东方A：2025 年年度报告", normalized.iloc[0]["Title"])
        self.assertEqual("巨潮资讯", normalized.iloc[0]["Source"])

    def test_announcement_frame_to_items_should_skip_empty_title(self):
        """空标题公告没有分析价值，应跳过。"""
        frame = pd.DataFrame(
            {
                "Symbol": ["000725", "000725"],
                "StockName": ["京东方A", "京东方A"],
                "Title": ["", "京东方A：董事会决议公告"],
                "PublishTime": ["", "2026-06-18"],
                "Source": ["巨潮资讯", "巨潮资讯"],
                "Url": ["", "http://example.com"],
            }
        )

        items = announcement_frame_to_items(frame)

        self.assertEqual(1, len(items))
        self.assertEqual("京东方A：董事会决议公告", items[0].title)

    def test_render_stock_announcements_text(self):
        """公告渲染文本应强调这是正式披露材料。"""
        frame = pd.DataFrame(
            {
                "Symbol": ["000725"],
                "StockName": ["京东方A"],
                "Title": ["京东方A：风险提示公告"],
                "PublishTime": ["2026-06-18"],
                "Source": ["巨潮资讯"],
                "Url": ["http://example.com"],
            }
        )
        text = render_stock_announcements_text(announcement_frame_to_items(frame))

        self.assertIn("上市公司正式披露材料", text)
        self.assertIn("京东方A：风险提示公告", text)
        self.assertIn("巨潮资讯", text)

    def test_date_helpers(self):
        """日期辅助函数应输出 AKShare 需要的格式。"""
        self.assertEqual("20260618", to_akshare_date("2026-06-18"))
        self.assertEqual("2026-03-20", calculate_announcement_start_date("2026-06-18", 90))

    def test_announcement_tool_should_be_registered(self):
        """公告工具应加入工具注册表。"""
        tool_names = [tool.name for tool in get_announcement_tools()]
        self.assertIn("get_stock_announcements_tool", tool_names)
        self.assertIn("get_stock_announcements_tool", get_tool_map())


if __name__ == "__main__":
    unittest.main()
