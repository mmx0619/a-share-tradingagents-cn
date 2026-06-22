import unittest

from tradingagents_cn.agents import (
    ResearchPlan,
    PortfolioRating,
    build_fallback_research_plan,
)
from tradingagents_cn.llm.structured_output import call_structured_output


class FakeStructuredClient:
    """测试用模型客户端。"""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def chat(self, messages, tools=None, tool_choice=None, temperature=0.0):
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


class StructuredOutputRetryTest(unittest.TestCase):
    def test_call_structured_output_should_retry_after_invalid_json(self):
        """第一次不是合法 JSON 时，应反馈错误并重试。"""
        client = FakeStructuredClient(
            [
                "我觉得可以买，但我不输出 JSON。",
                '{"recommendation":"Buy","rationale":"多头证据更强","strategic_actions":"小仓位试探"}',
            ]
        )

        result = call_structured_output(
            llm_client=client,
            messages=[{"role": "user", "content": "输出 ResearchPlan JSON"}],
            schema_model=ResearchPlan,
            fallback_factory=build_fallback_research_plan,
            max_retries=2,
        )

        self.assertFalse(result.used_fallback)
        self.assertEqual(2, result.attempts)
        self.assertEqual(PortfolioRating.BUY, result.value.recommendation)
        self.assertEqual(2, client.calls)
        self.assertTrue(any("没有通过程序校验" in msg["content"] for msg in result.messages if msg["role"] == "user"))

    def test_call_structured_output_should_return_fallback_after_retries(self):
        """多次失败后，应返回合法兜底对象。"""
        client = FakeStructuredClient(
            [
                "不是 JSON",
                '{"recommendation":"强烈看涨","rationale":"x","strategic_actions":"y"}',
                "还是不是 JSON",
            ]
        )

        result = call_structured_output(
            llm_client=client,
            messages=[{"role": "user", "content": "输出 ResearchPlan JSON"}],
            schema_model=ResearchPlan,
            fallback_factory=build_fallback_research_plan,
            max_retries=2,
        )

        self.assertTrue(result.used_fallback)
        self.assertEqual(3, result.attempts)
        self.assertEqual(PortfolioRating.HOLD, result.value.recommendation)
        self.assertIn("保守兜底", result.value.rationale)


if __name__ == "__main__":
    unittest.main()
