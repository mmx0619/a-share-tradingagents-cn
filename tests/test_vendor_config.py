import unittest

from tradingagents_cn.dataflows.vendor_config import (
    filter_sentiment_analyst,
    normalize_selected_analysts,
    normalize_vendor_overrides,
)


class VendorConfigTest(unittest.TestCase):
    def test_normalize_selected_analysts_should_support_aliases(self):
        """selected_analysts 应支持英文别名并保持顺序。"""
        analysts = normalize_selected_analysts("technical,social,news,fundamental")

        self.assertEqual(("market", "sentiment", "news", "fundamentals"), analysts)

    def test_filter_sentiment_analyst_should_remove_sentiment_when_disabled(self):
        """关闭情绪开关时，应从 Analyst 列表里移除 sentiment。"""
        analysts = filter_sentiment_analyst(
            ("market", "sentiment", "news"),
            include_sentiment=False,
        )

        self.assertEqual(("market", "news"), analysts)

    def test_normalize_vendor_overrides_should_parse_key_value_items(self):
        """命令行 vendor 覆盖项应解析成字典。"""
        overrides = normalize_vendor_overrides(
            ["market_data=akshare", "sentiment_sources=eastmoney,xueqiu"]
        )

        self.assertEqual("akshare", overrides["market_data"])
        self.assertEqual("eastmoney,xueqiu", overrides["sentiment_sources"])

    def test_normalize_vendor_overrides_should_support_human_friendly_aliases(self):
        """用户写 announcements 时，应映射到主图内部使用的 announcement_tool。"""
        overrides = normalize_vendor_overrides(["announcements=cninfo"])

        self.assertEqual("cninfo", overrides["announcement_tool"])
        self.assertNotIn("announcements", overrides)


if __name__ == "__main__":
    unittest.main()
