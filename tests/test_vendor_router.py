import unittest
from unittest.mock import patch

import pandas as pd

from tradingagents_cn.dataflows.realtime_quote import RealtimeQuote
from tradingagents_cn.dataflows.vendor_router import (
    UnsupportedVendorError,
    ensure_supported_vendor,
    list_supported_vendors,
    normalize_vendor_name,
    route_daily_history,
    route_realtime_quote,
    route_sentiment_items,
    route_stock_announcements,
)


class VendorRouterTest(unittest.TestCase):
    def test_should_normalize_vendor_alias(self):
        """vendor 别名应该先归一化，后续路由才不会散落在各个工具里。"""
        self.assertEqual("eastmoney", normalize_vendor_name("em"))
        self.assertEqual("sina", normalize_vendor_name("sina"))
        self.assertEqual("auto", normalize_vendor_name(""))

    def test_should_reject_unsupported_vendor(self):
        """不支持的数据源要明确报错，不能悄悄走到错误来源。"""
        with self.assertRaises(UnsupportedVendorError):
            ensure_supported_vendor("realtime_quote", "unknown_vendor")

    def test_realtime_quote_should_route_to_sina(self):
        """实时行情选择 sina 时，应该调用新浪行情入口。"""
        quote = RealtimeQuote(
            symbol="000725",
            name="京东方A",
            latest_price=5.0,
            change_amount=0.1,
            change_pct=2.0,
            open_price=4.9,
            previous_close=4.9,
            high_price=5.1,
            low_price=4.8,
            volume=1000000,
            amount=5000000,
            turnover_rate=None,
            volume_ratio=None,
            update_time="2026-06-20 10:00:00",
            source="sina",
        )

        with patch(
            "tradingagents_cn.dataflows.vendor_router.get_realtime_quote_from_sina",
            return_value=quote,
        ) as mocked_fetcher:
            result = route_realtime_quote("000725", vendor="sina")

        mocked_fetcher.assert_called_once_with("000725")
        self.assertEqual("sina", result.source)

    def test_daily_history_should_route_to_tencent(self):
        """历史日线选择 tencent 时，应该调用腾讯日线入口。"""
        frame = pd.DataFrame(
            [
                {
                    "Date": "2026-06-20",
                    "Open": 1.0,
                    "Close": 1.1,
                    "High": 1.2,
                    "Low": 0.9,
                    "Volume": 1000,
                    "Amount": 1100,
                }
            ]
        )

        with patch(
            "tradingagents_cn.dataflows.vendor_router.get_daily_history_from_tencent",
            return_value=frame,
        ) as mocked_fetcher:
            result = route_daily_history(
                "000725",
                start_date="2026-01-01",
                end_date="2026-06-20",
                vendor="tencent",
            )

        mocked_fetcher.assert_called_once_with("000725", "2026-01-01", "2026-06-20")
        self.assertEqual(["2026-06-20"], result["Date"].tolist())

    def test_announcements_should_route_to_cninfo(self):
        """公司公告选择 cninfo 时，应该调用巨潮公告入口并转成统一对象。"""
        frame = pd.DataFrame(
            [
                {
                    "securityCode": "000725",
                    "securityName": "京东方A",
                    "announcementTitle": "2026 年半年度报告",
                    "notice_date": "2026-06-20",
                    "url": "https://example.com/report",
                }
            ]
        )

        with patch(
            "tradingagents_cn.dataflows.vendor_router.fetch_cninfo_announcements",
            return_value=frame,
        ) as mocked_fetcher:
            result = route_stock_announcements(
                "000725",
                end_date="2026-06-20",
                lookback_days=30,
                max_items=1,
                vendor="cninfo",
            )

        mocked_fetcher.assert_called_once()
        self.assertEqual(1, len(result))
        self.assertEqual("000725", result[0].symbol)
        self.assertEqual("2026 年半年度报告", result[0].title)

    def test_sentiment_vendor_should_choose_default_source(self):
        """情绪面选择 xueqiu 时，默认只请求雪球这个来源。"""
        with patch(
            "tradingagents_cn.dataflows.vendor_router.get_stock_sentiment_items",
            return_value=[],
        ) as mocked_fetcher:
            route_sentiment_items("000725", max_items=3, vendor="xueqiu")

        mocked_fetcher.assert_called_once()
        self.assertEqual("xueqiu", mocked_fetcher.call_args.kwargs["sources"])

    def test_should_list_supported_vendors(self):
        """能力表要暴露出来，方便 CLI、文档和测试查看当前支持范围。"""
        vendors = list_supported_vendors()
        self.assertIn("realtime_quote", vendors)
        self.assertIn("sina", vendors["realtime_quote"])
        self.assertIn("cninfo", vendors["announcements"])


if __name__ == "__main__":
    unittest.main()
