import unittest

from tradingagents_cn.graph.research_report_pipeline import run_bull_bear_debate


class FakeDebateClient:
    """多空辩论测试用模型客户端。"""

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


class DebateStructuredTest(unittest.TestCase):
    def test_run_bull_bear_debate_should_use_structured_arguments(self):
        """多空辩论应输出可校验结构，再渲染成 Research Manager 可读文本。"""
        client = FakeDebateClient(
            [
                """
                {
                  "role": "bull",
                  "stance_strength": "strong",
                  "thesis": "多头认为技术面和基本面材料存在向上修复线索。",
                  "supporting_evidence": ["技术面报告显示趋势有改善线索", "基本面报告没有发现重大恶化"],
                  "opponent_rebuttals": ["当前暂无空头观点，因此先建立正方证据链"],
                  "uncertainties": ["新闻催化持续性仍需观察"],
                  "investment_implication": "Research Manager 可以提高正面材料权重，但仍需等待空头反驳。",
                  "debate_argument": "多头观点认为，现有材料支持继续观察向上修复机会。"
                }
                """,
                """
                {
                  "role": "bear",
                  "stance_strength": "medium",
                  "thesis": "空头认为上涨证据还不够充分，短线风险不能忽视。",
                  "supporting_evidence": ["新闻材料缺少强催化", "技术面仍可能出现反复"],
                  "opponent_rebuttals": ["多头对趋势修复的判断需要更多成交和价格确认"],
                  "uncertainties": ["如果后续放量突破，空头观点需要下调权重"],
                  "investment_implication": "Research Manager 应要求更明确的入场条件和风险边界。",
                  "debate_argument": "空头观点认为，当前材料不足以支持过度乐观。"
                }
                """,
            ]
        )

        result = run_bull_bear_debate(
            llm_client=client,
            symbol="000725",
            trade_date="2026-06-18",
            market_report="技术面报告。",
            news_report="新闻面报告。",
            fundamentals_report="基本面报告。",
            summary_report="综合报告。",
            max_debate_rounds=1,
        )

        self.assertEqual(2, client.calls)
        self.assertIn("Bull Researcher Round 1", result.debate_history)
        self.assertIn("Bear Researcher Round 1", result.debate_history)
        self.assertIn("**Debate Role**: bull", result.debate_history)
        self.assertIn("**Stance Strength**: medium", result.debate_history)
        self.assertIn("bull_round_1", result.messages_by_agent)
        self.assertEqual(3, len(result.messages_by_agent["bear_round_1"]))


if __name__ == "__main__":
    unittest.main()
