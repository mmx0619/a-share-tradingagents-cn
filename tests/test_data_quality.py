"""A 股数据质量校验测试。

这些测试不联网。

它们只检查数据质量规则是否能发现明显问题：
1. 历史行情太少；
2. 历史行情过旧；
3. 实时行情缺关键字段；
4. 新闻过旧；
5. 新闻日期无法解析。
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import pandas as pd

from tradingagents_cn.dataflows.data_quality import (
    parse_datetime_safely,
    render_data_quality_issues,
    validate_announcement_items,
    validate_daily_history,
    validate_fundamental_texts,
    validate_realtime_quote,
    validate_sentiment_items,
    validate_stock_news_items,
)


@dataclass
class FakeQuote:
    """测试用实时行情对象。"""

    latest_price: float | None = None
    previous_close: float | None = None
    amount: float | None = None
    volume: float | None = None
    update_time: str | None = None


@dataclass
class FakeNewsItem:
    """测试用新闻对象。"""

    publish_time: str


@dataclass
class FakeAnnouncementItem:
    """测试用公告对象。"""

    publish_time: str


@dataclass
class FakeSentimentItem:
    """测试用情绪材料对象。"""

    source: str = ""
    title: str = ""


class DataQualityTest(unittest.TestCase):
    """测试 A 股输入数据质量校验。"""

    def test_daily_history_should_warn_when_rows_are_too_few(self) -> None:
        """历史行情行数不足时，应该提示技术指标可靠性下降。"""
        history = pd.DataFrame(
            {
                "Date": ["2026-06-17", "2026-06-18"],
                "Close": [10.0, 10.2],
            }
        )

        issues = validate_daily_history(
            history,
            trade_date="2026-06-18",
            min_rows=120,
        )

        self.assertTrue(any("历史行情数据较少" in issue for issue in issues))

    def test_daily_history_should_warn_when_latest_date_is_stale(self) -> None:
        """历史行情最新日期距离分析日期太远时，应该提示数据过旧。"""
        history = pd.DataFrame(
            {
                "Date": ["2026-05-01", "2026-05-02"],
                "Close": [10.0, 10.2],
            }
        )

        issues = validate_daily_history(
            history,
            trade_date="2026-06-18",
            min_rows=1,
            stale_days=10,
        )

        self.assertTrue(any("可能存在停牌" in issue for issue in issues))

    def test_realtime_quote_should_warn_when_key_fields_missing(self) -> None:
        """实时行情缺最新价、昨收和成交信息时，应该提示。"""
        issues = validate_realtime_quote(FakeQuote())

        self.assertTrue(any("缺少最新价" in issue for issue in issues))
        self.assertTrue(any("缺少有效昨收价" in issue for issue in issues))
        self.assertTrue(any("缺少成交量和成交额" in issue for issue in issues))
        self.assertTrue(any("缺少更新时间" in issue for issue in issues))

    def test_stock_news_should_warn_when_news_is_stale(self) -> None:
        """最近新闻过旧时，应该提示新闻材料可能偏旧。"""
        items = [FakeNewsItem(publish_time="2026-05-01 10:00:00")]

        issues = validate_stock_news_items(
            items,
            trade_date="2026-06-18",
            stale_days=14,
        )

        self.assertTrue(any("新闻材料可能偏旧" in issue for issue in issues))

    def test_stock_news_should_warn_when_time_unparseable(self) -> None:
        """新闻时间都无法解析时，应该提示。"""
        items = [FakeNewsItem(publish_time="不是日期")]

        issues = validate_stock_news_items(items, trade_date="2026-06-18")

        self.assertTrue(any("发布时间无法解析" in issue for issue in issues))

    def test_parse_datetime_safely(self) -> None:
        """常见日期格式应该能被解析。"""
        self.assertIsNotNone(parse_datetime_safely("2026-06-18 09:30:00"))
        self.assertIsNotNone(parse_datetime_safely("2026/06/18"))
        self.assertIsNone(parse_datetime_safely(""))

    def test_announcement_should_warn_when_empty(self) -> None:
        """公告为空时，应提示正式披露材料缺失。"""
        issues = validate_announcement_items([], trade_date="2026-06-18")

        self.assertTrue(any("公司公告为空" in issue for issue in issues))

    def test_sentiment_should_warn_when_fields_missing(self) -> None:
        """情绪材料缺标题或来源时，应提示。"""
        issues = validate_sentiment_items([FakeSentimentItem()])

        self.assertTrue(any("缺少标题" in issue for issue in issues))
        self.assertTrue(any("缺少来源" in issue for issue in issues))

    def test_fundamental_texts_should_warn_when_short(self) -> None:
        """基本面文本过短时，应提示材料可能异常。"""
        issues = validate_fundamental_texts({"利润表": "太短"})

        self.assertTrue(any("文本较短" in issue for issue in issues))

    def test_render_data_quality_issues(self) -> None:
        """数据质量问题应渲染成工具可读文本。"""
        text = render_data_quality_issues(["新闻材料可能偏旧。"])

        self.assertIn("数据质量提示", text)
        self.assertIn("新闻材料可能偏旧", text)


if __name__ == "__main__":
    unittest.main()
