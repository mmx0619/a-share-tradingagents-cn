"""最终报告输出规则测试。

这些测试不调用真实行情、不调用大模型、不访问网络。

它们只检查“输出层”的固定规则：

1. 英文交易动作和评级，能不能翻译成用户看得懂的中文结论。
2. 报告文件名，能不能使用“股票名_股票代码_日期”的格式。

为什么要测这个？
    因为用户最终看到的是报告和控制台回答。
    即使前面的 Agent 都跑对了，
    如果最终输出仍然只写 Sell / Underweight，
    人还是会觉得“不知道到底该买还是卖”。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

from tradingagents_cn.graph.display_text import (
    localize_report_text,
    translate_machine_action,
    translate_paper_status,
)
from tradingagents_cn.graph.final_report import (
    build_operation_sentence,
    build_trading_status,
    translate_portfolio_rating,
    translate_trader_action,
)
from tradingagents_cn.graph.user_question_pipeline import (
    build_single_stock_answer,
    build_report_path,
    sanitize_filename_part,
)


class ReportOutputRulesTest(unittest.TestCase):
    """测试最终报告的人类可读输出规则。"""

    def test_sell_and_underweight_should_be_clear_sell_conclusion(self) -> None:
        """Sell + Underweight 必须明确表达“不建议买入 / 减仓或卖出”。

        这是用户这次指出的问题：
            最终报告里虽然有 Sell，
            但普通人第一眼不一定知道该怎么操作。
        """
        conclusion = build_operation_sentence(
            trader_action="Sell",
            portfolio_rating="Underweight",
        )

        self.assertIn("不建议买入", conclusion)
        self.assertIn("减仓或卖出", conclusion)

    def test_buy_and_overweight_should_be_clear_buy_conclusion(self) -> None:
        """Buy + Overweight 必须表达可以考虑买入或加仓。"""
        conclusion = build_operation_sentence(
            trader_action="Buy",
            portfolio_rating="Overweight",
        )

        self.assertIn("买入或加仓", conclusion)

    def test_hold_and_hold_should_be_clear_wait_conclusion(self) -> None:
        """Hold + Hold 必须表达观望，不要写成模糊废话。"""
        conclusion = build_operation_sentence(
            trader_action="Hold",
            portfolio_rating="Hold",
        )

        self.assertIn("观望", conclusion)

    def test_rating_and_action_translation(self) -> None:
        """英文枚举值必须能翻译成中文。"""
        self.assertEqual(translate_trader_action("Buy"), "买入")
        self.assertEqual(translate_trader_action("Sell"), "卖出或回避")
        self.assertEqual(translate_machine_action("SELL"), "卖出或回避")
        self.assertEqual(translate_portfolio_rating("Underweight"), "低配，偏谨慎")
        self.assertEqual(translate_portfolio_rating("Overweight"), "增配，偏积极")
        self.assertEqual(translate_paper_status("filled"), "已成交")

    def test_fixed_report_labels_should_be_localized(self) -> None:
        """最终报告里的固定英文角色名要翻译成中文。"""
        text = localize_report_text(
            "Portfolio Manager reads Trader Agent output and returns Underweight."
        )

        self.assertIn("组合经理", text)
        self.assertIn("交易员", text)
        self.assertIn("低配，偏谨慎", text)
        self.assertNotIn("Portfolio Manager", text)
        self.assertNotIn("Trader Agent", text)

    def test_risk_debate_labels_should_be_localized(self) -> None:
        """风险辩论报告里的英文标题和枚举值要翻译成中文。"""
        text = localize_report_text(
            "\n".join(
                [
                    "Conservative Risk Analyst Round 1: **Risk Role**: conservative",
                    "**Risk Level**: high",
                    "**Allow Trade**: 不建议继续执行交易提案",
                    "",
                    "**Risk Triggers**:",
                    "- 跌破 10 日均线",
                    "",
                    "**Mitigation Plan**: 当前不新建多头仓位。",
                    "",
                    "**Position Sizing Advice**: 建议将风险敞口降至 0。",
                    "",
                    "**Debate Argument**: 当前空头证据更强。",
                ]
            )
        )

        self.assertIn("保守风险分析员 第 1 轮", text)
        self.assertIn("**风险角色**", text)
        self.assertIn("保守风险分析员", text)
        self.assertIn("**风险等级**", text)
        self.assertIn("高", text)
        self.assertIn("**是否允许交易**", text)
        self.assertIn("**风险触发条件**", text)
        self.assertIn("**风险缓释方案**", text)
        self.assertIn("**仓位建议**", text)
        self.assertIn("**辩论观点**", text)
        self.assertNotIn("Risk Triggers", text)
        self.assertNotIn("Mitigation Plan", text)
        self.assertNotIn("Position Sizing Advice", text)
        self.assertNotIn("Debate Argument", text)
        self.assertNotIn("Round 1", text)
        self.assertNotIn("conservative", text)

    def test_bull_bear_debate_labels_should_be_localized(self) -> None:
        """多空辩论报告里的英文标题和枚举值要翻译成中文。"""
        text = localize_report_text(
            "\n".join(
                [
                    "Bull Researcher Round 1: **Debate Role**: bull",
                    "**Stance Strength**: strong",
                    "**Thesis**: 多头核心观点。",
                    "**Supporting Evidence**:",
                    "- 技术面改善",
                    "**Opponent Rebuttals**:",
                    "- 反驳空头观点",
                    "**Uncertainties**:",
                    "- 仍有不确定性",
                    "**Investment Implication**: 可提高正面材料权重。",
                ]
            )
        )

        self.assertIn("多头研究员 第 1 轮", text)
        self.assertIn("**多空角色**", text)
        self.assertIn("**观点强度**", text)
        self.assertIn("强", text)
        self.assertIn("**核心论点**", text)
        self.assertIn("**支持证据**", text)
        self.assertIn("**对方观点反驳**", text)
        self.assertIn("**不确定性**", text)
        self.assertIn("**投资含义**", text)
        self.assertNotIn("Debate Role", text)
        self.assertNotIn("Stance Strength", text)
        self.assertNotIn("Round 1", text)
        self.assertNotIn("bull", text)

    def test_common_technical_terms_should_be_localized(self) -> None:
        """常见技术指标英文缩写要尽量翻译成中文。"""
        text = localize_report_text("股价跌破10日EMA，MACD和RSI转弱，MFI背离。")

        self.assertIn("10日指数移动平均线", text)
        self.assertIn("指数平滑异同移动平均线", text)
        self.assertIn("相对强弱指标", text)
        self.assertIn("资金流量指标", text)
        self.assertNotIn("EMA", text)
        self.assertNotIn("MACD", text)
        self.assertNotIn("RSI", text)
        self.assertNotIn("MFI", text)

    def test_single_stock_answer_should_hide_machine_english_enums(self) -> None:
        """控制台短回答里不要再出现 Sell / Underweight 等机器枚举。"""
        route = SimpleNamespace(stock_name="天娱数科", symbol="002354")
        result = SimpleNamespace(
            trader_proposal=SimpleNamespace(
                action=SimpleNamespace(value="Sell"),
                position_sizing="暂不新增仓位。",
            ),
            portfolio_decision=SimpleNamespace(
                rating=SimpleNamespace(value="Underweight"),
                executive_summary="基本面和风险项偏弱。",
            ),
            research_plan=SimpleNamespace(
                recommendation=SimpleNamespace(value="Underweight"),
            ),
            trade_signal=SimpleNamespace(
                action="SELL",
                chinese_action="降低仓位",
            ),
            risk_guardrail=SimpleNamespace(
                chinese_summary="不允许新增仓位。",
            ),
            paper_trading_result=None,
            full_state_log_path=None,
        )

        answer = build_single_stock_answer(
            route=route,
            result=result,
            report_path=Path("outputs/user_questions/天娱数科_002354_2026-06-22_final_report.md"),
        )

        self.assertIn("交易员动作：卖出或回避", answer)
        self.assertIn("交易状态：减仓风控", answer)
        self.assertIn("现在能不能买：现在不新增仓位。", answer)
        self.assertIn("组合经理最终评级：低配，偏谨慎", answer)
        self.assertIn("机器交易信号：卖出或回避，降低仓位", answer)
        self.assertNotIn("Sell", answer)
        self.assertNotIn("SELL", answer)
        self.assertNotIn("Underweight", answer)
        self.assertNotIn("Portfolio Manager", answer)

    def test_strong_but_overheated_stock_should_wait_for_pullback(self) -> None:
        """基本面强但短线过热时，应显示等待回调，而不是粗暴卖出。"""
        result = SimpleNamespace(
            trader_proposal=SimpleNamespace(
                action=SimpleNamespace(value="Sell"),
                position_sizing="暂不新增仓位。",
            ),
            portfolio_decision=SimpleNamespace(
                rating=SimpleNamespace(value="Underweight"),
                executive_summary="基本面强劲但技术面超买，短期涨幅过大，等待回调至均线支撑后再评估。",
                investment_thesis="行业龙头，营收增长，但当前不适合追高。",
            ),
            research_plan=SimpleNamespace(
                recommendation=SimpleNamespace(value="Underweight"),
            ),
            trade_signal=SimpleNamespace(
                action="SELL",
                chinese_action="降低仓位",
            ),
            risk_guardrail=SimpleNamespace(
                chinese_summary="不允许新增仓位。",
                allow_new_position=False,
                risk_band="blocked",
            ),
            investment_plan="等待回调确认。",
            trader_plan="不追高。",
            summary_report="基本面强劲，但超买。",
            market_report="股价突破布林带上轨。",
            news_report="暂无重大利空。",
            fundamentals_report="营收增长，行业龙头。",
        )

        status = build_trading_status(result)

        self.assertEqual("WAIT_PULLBACK", status.code)
        self.assertEqual("等待回调", status.label)
        self.assertIn("不追高", status.current_action)

    def test_severe_risk_stock_should_be_sell_avoid(self) -> None:
        """严重风险项明确时，应显示卖出回避。"""
        result = SimpleNamespace(
            trader_proposal=SimpleNamespace(action=SimpleNamespace(value="Sell")),
            portfolio_decision=SimpleNamespace(
                rating=SimpleNamespace(value="Underweight"),
                executive_summary="公司全年亏损，存在商誉减值和监管风险。",
                investment_thesis="基本面与股价严重脱节。",
            ),
            risk_guardrail=SimpleNamespace(allow_new_position=False, risk_band="blocked"),
            investment_plan="回避。",
            trader_plan="卖出。",
            summary_report="风险较高。",
            market_report="高位波动。",
            news_report="监管风险。",
            fundamentals_report="全年亏损，商誉减值。",
        )

        status = build_trading_status(result)

        self.assertEqual("SELL_AVOID", status.code)
        self.assertEqual("卖出回避", status.label)

    def test_report_path_should_include_stock_name_and_symbol(self) -> None:
        """报告文件名必须包含股票名和股票代码。"""
        path = build_report_path(
            output_dir="outputs/user_questions",
            stock_name="京东方A",
            symbol="000725",
            trade_date="2026-06-18",
        )

        self.assertEqual(
            str(path),
            "outputs\\user_questions\\京东方A_000725_2026-06-18_final_report.md",
        )

    def test_filename_part_should_remove_windows_invalid_chars(self) -> None:
        """股票名里如果混入 Windows 禁用字符，必须被替换。"""
        cleaned = sanitize_filename_part('京东方A/测试:版本*')

        self.assertEqual(cleaned, "京东方A_测试_版本_")


if __name__ == "__main__":
    unittest.main()
