import unittest

from tradingagents_cn.graph.research_report_pipeline import run_risk_debate


class FakeRiskDebateClient:
    """风险辩论测试用模型客户端。"""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def chat(self, messages, tools=None, tool_choice=None, temperature=0.0):
        """模拟真实聊天模型返回。"""
        self.calls += 1
        content = self.outputs.pop(0)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                    }
                }
            ]
        }


class RiskDebateStructuredTest(unittest.TestCase):
    def test_run_risk_debate_should_use_structured_assessments(self):
        """风险辩论三方应输出可校验结构，再渲染成给下游看的文本。"""
        client = FakeRiskDebateClient(
            [
                """
                {
                  "role": "aggressive",
                  "risk_level": "medium",
                  "allow_trade": true,
                  "key_risks": ["追高风险"],
                  "risk_triggers": ["跌破短期均线"],
                  "mitigation_plan": "只在条件满足时小仓位试探。",
                  "position_sizing_advice": "单笔仓位控制在 5% 以内。",
                  "debate_argument": "如果技术面和消息面继续共振，可以保留试探机会。"
                }
                """,
                """
                {
                  "role": "conservative",
                  "risk_level": "high",
                  "allow_trade": false,
                  "key_risks": ["波动放大", "回撤不可控"],
                  "risk_triggers": ["放量下跌"],
                  "mitigation_plan": "等待风险释放后再评估。",
                  "position_sizing_advice": "不新增仓位。",
                  "debate_argument": "当前更应优先控制回撤。"
                }
                """,
                """
                {
                  "role": "neutral",
                  "risk_level": "medium",
                  "allow_trade": true,
                  "key_risks": ["短线不确定性"],
                  "risk_triggers": ["跌破预设止损"],
                  "mitigation_plan": "设置止损并降低单次投入。",
                  "position_sizing_advice": "轻仓观察。",
                  "debate_argument": "可以保留机会，但必须把仓位和止损写清楚。"
                }
                """,
            ]
        )

        result = run_risk_debate(
            llm_client=client,
            symbol="000725",
            trade_date="2026-06-18",
            market_report="技术面报告。",
            news_report="新闻面报告。",
            fundamentals_report="基本面报告。",
            investment_plan="研究经理计划。",
            trader_plan="交易员提案。",
            max_risk_discuss_rounds=1,
        )

        self.assertEqual(3, client.calls)
        self.assertIn("Aggressive Risk Analyst Round 1", result.risk_history)
        self.assertIn("**Risk Level**: high", result.risk_history)
        self.assertIn("**Position Sizing Advice**", result.risk_history)
        self.assertIn("conservative_risk_round_1", result.messages_by_agent)
        self.assertEqual(3, len(result.messages_by_agent["neutral_risk_round_1"]))


if __name__ == "__main__":
    unittest.main()
