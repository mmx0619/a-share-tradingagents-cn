import unittest
from unittest.mock import Mock, patch

from tradingagents_cn.dataflows.sentiment import (
    clean_html_text,
    get_eastmoney_guba_items,
    parse_generic_discussion_html,
    parse_eastmoney_guba_html,
    parse_sentiment_sources,
    render_stock_sentiment_text,
    to_xueqiu_symbol,
)


class SentimentDataflowTest(unittest.TestCase):
    def test_parse_eastmoney_guba_html_should_extract_titles(self):
        """东方财富股吧 HTML 应解析出讨论标题和链接。"""
        html = """
        <html>
            <a href="/news,000725,123.html"> 京东方A今天放量了 </a>
            <a href="/news,000725,124.html"><span>面板周期怎么看</span></a>
            <a href="#">首页</a>
        </html>
        """

        items = parse_eastmoney_guba_html(html, base_url="https://guba.eastmoney.com")

        self.assertEqual(2, len(items))
        self.assertEqual("东方财富股吧", items[0].source)
        self.assertEqual("京东方A今天放量了", items[0].title)
        self.assertEqual(
            "https://guba.eastmoney.com/news,000725,123.html",
            items[0].url,
        )

    def test_clean_html_text_should_remove_tags(self):
        """清理函数应去掉标签和多余空白。"""
        self.assertEqual("面板 周期", clean_html_text("<span>面板</span>   周期"))

    def test_parse_sentiment_sources_should_support_cn_aliases(self):
        """情绪源配置应支持中文别名。"""
        sources = parse_sentiment_sources("东方财富股吧,雪球,同花顺股吧,淘股吧")

        self.assertEqual(["eastmoney", "xueqiu", "tonghuashun", "taoguba"], sources)

    def test_parse_generic_discussion_html_should_extract_titles(self):
        """通用解析器应能从普通 HTML 里提取标题。"""
        html = '<a href="/post/1">讨论大唐发电的分歧</a>'

        items = parse_generic_discussion_html(
            html,
            base_url="https://example.com",
            source="测试社区",
        )

        self.assertEqual(1, len(items))
        self.assertEqual("测试社区", items[0].source)
        self.assertEqual("讨论大唐发电的分歧", items[0].title)
        self.assertEqual("https://example.com/post/1", items[0].url)

    def test_to_xueqiu_symbol_should_add_market_prefix(self):
        """雪球代码应按 A 股市场添加前缀。"""
        self.assertEqual("SH600519", to_xueqiu_symbol("600519"))
        self.assertEqual("SZ000725", to_xueqiu_symbol("000725"))

    @patch("tradingagents_cn.dataflows.sentiment.requests.get")
    def test_get_eastmoney_guba_items_should_return_empty_on_request_error(self, mock_get):
        """股吧请求失败时返回空列表，不影响主流程。"""
        mock_get.side_effect = RuntimeError("network broken")

        items = get_eastmoney_guba_items("000725")

        self.assertEqual([], items)

    @patch("tradingagents_cn.dataflows.sentiment.requests.get")
    def test_get_eastmoney_guba_items_should_parse_response(self, mock_get):
        """请求成功时应解析页面标题。"""
        response = Mock()
        response.text = '<a href="/news,000725,123.html">京东方A讨论</a>'
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        items = get_eastmoney_guba_items("000725")

        self.assertEqual("京东方A讨论", items[0].title)

    def test_render_stock_sentiment_text_should_explain_missing_data(self):
        """没有情绪数据时，应明确说明这是数据缺口。"""
        text = render_stock_sentiment_text([])

        self.assertIn("暂未获取到", text)
        self.assertIn("不代表市场没有情绪", text)


if __name__ == "__main__":
    unittest.main()
