import os
import tempfile
import unittest
from unittest.mock import patch

from tradingagents_cn.dataflows.announcements import StockAnnouncementItem
from tradingagents_cn.dataflows.sentiment import SentimentItem
from tradingagents_cn.dataflows.stock_news import StockNewsItem
from tradingagents_cn.tools.announcement_tools import get_cached_stock_announcements_text
from tradingagents_cn.tools.news_tools import get_cached_stock_news_text
from tradingagents_cn.tools.sentiment_tools import get_cached_stock_sentiment_text


class MaterialToolCacheTest(unittest.TestCase):
    def test_news_tool_should_use_text_cache(self):
        """新闻工具第二次读取同一参数时，应命中本地文本缓存。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            calls = {"count": 0}

            def fake_get_news(symbol, max_items=5, **kwargs):
                calls["count"] += 1
                return [
                    StockNewsItem(
                        symbol=symbol,
                        title=f"新闻标题 {calls['count']}",
                        content="新闻正文",
                        publish_time="2026-06-18",
                        source="测试来源",
                        url="http://example.com/news",
                    )
                ]

            with patch.dict(os.environ, {"TRADINGAGENTS_CN_CACHE_DIR": temp_dir}):
                with patch(
                    "tradingagents_cn.tools.news_tools.route_stock_news",
                    side_effect=fake_get_news,
                ):
                    first = get_cached_stock_news_text("000725", max_items=1)
                    second = get_cached_stock_news_text("000725", max_items=1)

        self.assertIn("缓存状态：联网刷新", first)
        self.assertIn("缓存状态：本地缓存", second)
        self.assertIn("新闻标题 1", second)
        self.assertEqual(1, calls["count"])

    def test_announcement_tool_should_use_text_cache(self):
        """公告工具第二次读取同一参数时，应命中本地文本缓存。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            calls = {"count": 0}

            def fake_get_announcements(
                symbol,
                end_date,
                lookback_days=90,
                max_items=10,
                category="",
                **kwargs,
            ):
                calls["count"] += 1
                return [
                    StockAnnouncementItem(
                        symbol=symbol,
                        stock_name="京东方A",
                        title=f"公告标题 {calls['count']}",
                        publish_time=end_date,
                        source="巨潮资讯",
                        url="http://example.com/announcement",
                    )
                ]

            with patch.dict(os.environ, {"TRADINGAGENTS_CN_CACHE_DIR": temp_dir}):
                with patch(
                    "tradingagents_cn.tools.announcement_tools.route_stock_announcements",
                    side_effect=fake_get_announcements,
                ):
                    first = get_cached_stock_announcements_text(
                        "000725",
                        end_date="2026-06-18",
                        max_items=1,
                    )
                    second = get_cached_stock_announcements_text(
                        "000725",
                        end_date="2026-06-18",
                        max_items=1,
                    )

        self.assertIn("缓存状态：联网刷新", first)
        self.assertIn("缓存状态：本地缓存", second)
        self.assertIn("公告标题 1", second)
        self.assertEqual(1, calls["count"])

    def test_sentiment_tool_should_use_text_cache(self):
        """情绪工具第二次读取同一参数时，应命中本地文本缓存。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            calls = {"count": 0}

            def fake_get_sentiment(symbol, max_items=10, **kwargs):
                calls["count"] += 1
                return [
                    SentimentItem(
                        source="东方财富股吧",
                        title=f"情绪标题 {calls['count']}",
                        url="http://example.com/guba",
                    )
                ]

            with patch.dict(os.environ, {"TRADINGAGENTS_CN_CACHE_DIR": temp_dir}):
                with patch(
                    "tradingagents_cn.tools.sentiment_tools.route_sentiment_items",
                    side_effect=fake_get_sentiment,
                ):
                    first = get_cached_stock_sentiment_text("000725", max_items=1)
                    second = get_cached_stock_sentiment_text("000725", max_items=1)

        self.assertIn("缓存状态：联网刷新", first)
        self.assertIn("缓存状态：本地缓存", second)
        self.assertIn("情绪标题 1", second)
        self.assertEqual(1, calls["count"])


if __name__ == "__main__":
    unittest.main()
