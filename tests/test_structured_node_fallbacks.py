import unittest

from tradingagents_cn.agents import (
    PortfolioRating,
    TraderAction,
    build_fallback_debate_argument,
    build_fallback_portfolio_decision,
    build_fallback_research_plan,
    build_fallback_risk_assessment,
    build_fallback_trader_proposal,
    render_debate_argument,
    render_risk_assessment,
)
from tradingagents_cn.router.llm_question_router import build_fallback_llm_route_decision


class StructuredNodeFallbacksTest(unittest.TestCase):
    def test_research_manager_fallback_should_be_hold(self):
        """Research Manager 兜底必须是合法 Hold。"""
        plan = build_fallback_research_plan("bad json")

        self.assertEqual(PortfolioRating.HOLD, plan.recommendation)
        self.assertIn("保守兜底", plan.rationale)

    def test_trader_fallback_should_be_hold(self):
        """Trader 兜底必须是合法 Hold。"""
        proposal = build_fallback_trader_proposal("bad json")

        self.assertEqual(TraderAction.HOLD, proposal.action)
        self.assertIsNone(proposal.entry_price)
        self.assertIsNone(proposal.stop_loss)

    def test_portfolio_fallback_should_be_hold(self):
        """Portfolio Manager 兜底必须是合法 Hold。"""
        decision = build_fallback_portfolio_decision("bad json")

        self.assertEqual(PortfolioRating.HOLD, decision.rating)
        self.assertIsNone(decision.price_target)

    def test_risk_fallback_should_block_trade(self):
        """风险辩论兜底必须是合法 high risk，并且不允许继续交易。"""
        assessment = build_fallback_risk_assessment("aggressive", "bad json")
        rendered = render_risk_assessment(assessment)

        self.assertEqual("aggressive", assessment.role)
        self.assertEqual("high", assessment.risk_level)
        self.assertFalse(assessment.allow_trade)
        self.assertIn("不建议继续执行交易提案", rendered)

    def test_debate_fallback_should_be_low_confidence(self):
        """多空辩论兜底必须是合法 weak 观点，避免污染 Research Manager。"""
        argument = build_fallback_debate_argument("bull", "bad json")
        rendered = render_debate_argument(argument)

        self.assertEqual("bull", argument.role)
        self.assertEqual("weak", argument.stance_strength)
        self.assertIn("低置信度", rendered)

    def test_router_fallback_should_be_unknown_low_confidence(self):
        """Router 兜底必须是合法 unknown/low。"""
        decision = build_fallback_llm_route_decision("bad json")

        self.assertEqual("unknown", decision.intent)
        self.assertEqual("low", decision.confidence)


if __name__ == "__main__":
    unittest.main()
