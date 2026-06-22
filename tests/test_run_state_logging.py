import json
import tempfile
import unittest
from pathlib import Path

from tradingagents_cn.graph.run_state_logging import save_full_state_json


class FakeResult:
    """给 full_state 测试用的最小结果对象。"""

    selected_analysts = ("market", "sentiment")
    market_report = "技术面报告"
    sentiment_report = "情绪面报告"
    news_report = "新闻面报告"
    fundamentals_report = "基本面报告"
    summary_report = "综合报告"
    max_debate_rounds = 1
    bull_argument = "多头观点"
    bear_argument = "空头观点"
    debate_history = "多空辩论"
    research_plan = {"recommendation": "Hold"}
    investment_plan = "研究计划"
    trader_proposal = {"action": "Hold"}
    trader_plan = "交易计划"
    portfolio_decision = {"rating": "Hold"}
    final_trade_decision = "最终决策"
    trade_signal = {"action": "HOLD"}
    risk_guardrail = {"allow_new_position": False}
    risk_debate_history = "风险辩论"
    aggressive_risk_argument = "激进风险观点"
    conservative_risk_argument = "保守风险观点"
    neutral_risk_argument = "中性风险观点"
    messages_by_agent = {
        "market": [
            {"role": "user", "content": "test"},
            {
                "role": "tool",
                "tool_call_id": "tool-1",
                "content": "数据质量提示：\n- 新闻材料可能偏旧。\n\n工具正文",
            },
        ]
    }
    tool_call_trace = [
        {
            "agent": "market",
            "event": "assistant_tool_call",
            "tool_name": "get_market_technical_snapshot",
        }
    ]
    tool_call_stats = {
        "total_tool_calls": 1,
        "by_agent": {
            "market": {
                "tool_call_count": 1,
                "tool_names": ["get_market_technical_snapshot"],
            }
        },
    }
    reflection_summary = {
        "benchmark_symbol": "000300",
        "benchmark_name": "沪深300",
        "updated_pending_count": 1,
    }
    paper_trading_result = {
        "enabled": True,
        "status": "filled",
        "order": {"action": "BUY", "shares": 100},
    }
    data_errors = []

    @property
    def final_state(self):
        return {
            "symbol": "000725",
            "trade_date": "2026-06-18",
        }


class RunStateLoggingTest(unittest.TestCase):
    def test_save_full_state_json_should_write_key_fields(self):
        """full_state.json 应保存报告、决策和 messages_by_agent。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_full_state_json(FakeResult(), output_dir=temp_dir)
            payload = json.loads(Path(path).read_text(encoding="utf-8"))

        self.assertTrue(str(path).endswith("full_state.json"))
        self.assertEqual("000725", payload["metadata"]["symbol"])
        self.assertEqual(["market", "sentiment"], payload["metadata"]["selected_analysts"])
        self.assertEqual("情绪面报告", payload["reports"]["sentiment_report"])
        self.assertIn("market", payload["messages_by_agent"])
        self.assertEqual(1, payload["tool_call_stats"]["total_tool_calls"])
        self.assertEqual("沪深300", payload["reflection_summary"]["benchmark_name"])
        self.assertEqual("filled", payload["paper_trading"]["status"])
        self.assertTrue(payload["data_quality_issues"])


if __name__ == "__main__":
    unittest.main()
