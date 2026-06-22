"""A 股选股/筛股数据层和 Agent 测试。

这些测试不联网，不调用真实大模型。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from tradingagents_cn.agents.stock_screening_agent import build_stock_screening_agent_prompt
from tradingagents_cn.dataflows.stock_screening import (
    StockScreeningConfig,
    apply_light_fundamental_filter,
    enrich_candidates_with_hot_sectors,
    fetch_with_retries,
    filter_screening_candidates,
    get_stock_screening_candidates,
    normalize_spot_frame,
    rank_screening_candidates,
    render_stock_screening_text,
)
from tradingagents_cn.graph.user_question_pipeline import run_stock_screening_answer
from tradingagents_cn.graph.stock_screening_deep_pipeline import (
    DeepScreeningItem,
    DeepScreeningResult,
)


class FakeLLMClient:
    """测试用成功模型客户端。"""

    def chat(self, messages, tools=None, tool_choice=None, temperature=0.2):
        return {
            "choices": [
                {
                    "message": {
                        "content": "候选观察名单：半导体A、机器人B。仅供继续研究，不是直接买入建议。"
                    }
                }
            ]
        }


class BrokenLLMClient:
    """测试用失败模型客户端。"""

    def chat(self, messages, tools=None, tool_choice=None, temperature=0.2):
        raise RuntimeError("模型超时")


def fake_candidates() -> pd.DataFrame:
    """构造测试用候选池。"""
    return pd.DataFrame(
        {
            "Symbol": ["000001", "000002"],
            "Name": ["半导体A", "机器人B"],
            "Latest": [10.0, 20.0],
            "ChangePct": [5.0, 4.0],
            "Amount": [500_000_000, 400_000_000],
            "TurnoverRate": [8.0, 6.0],
            "VolumeRatio": [2.0, 1.5],
        }
    )


class StockScreeningTest(unittest.TestCase):
    """测试选股/筛股能力。"""

    def test_normalize_spot_frame(self) -> None:
        """全市场快照应整理成稳定字段。"""
        raw = pd.DataFrame(
            {
                "代码": ["000001", "600000"],
                "名称": ["平安银行", "浦发银行"],
                "最新价": [10.0, 8.0],
                "涨跌幅": [1.0, -1.0],
                "成交额": [200_000_000, 150_000_000],
                "换手率": [2.0, 1.0],
                "量比": [1.2, 0.8],
                "市盈率-动态": [12.5, 20.0],
                "市净率": [0.8, 1.1],
                "总市值": [300_000_000_000, 250_000_000_000],
                "流通市值": [250_000_000_000, 200_000_000_000],
            }
        )

        normalized = normalize_spot_frame(raw)

        self.assertEqual(normalized.iloc[0]["Symbol"], "000001")
        self.assertEqual(normalized.iloc[1]["Symbol"], "600000")
        self.assertIn("ChangePct", normalized.columns)
        self.assertEqual(normalized.iloc[0]["DynamicPE"], 12.5)
        self.assertEqual(normalized.iloc[1]["TotalMarketCap"], 250_000_000_000)

    def test_filter_candidates_should_remove_st_and_low_amount(self) -> None:
        """过滤阶段应排除 ST、低成交额、低价股。"""
        data = pd.DataFrame(
            {
                "Symbol": ["000001", "000002", "000003"],
                "Name": ["正常A", "ST风险", "低成交"],
                "Latest": [10.0, 10.0, 10.0],
                "ChangePct": [3.0, 5.0, 4.0],
                "Amount": [200_000_000, 300_000_000, 1_000_000],
                "TurnoverRate": [2.0, 3.0, 4.0],
                "VolumeRatio": [1.0, 1.0, 1.0],
                "DynamicPE": [30.0, 40.0, 50.0],
                "TotalMarketCap": [10_000_000_000, 10_000_000_000, 10_000_000_000],
            }
        )

        filtered = filter_screening_candidates(data, StockScreeningConfig())

        self.assertEqual(filtered["Name"].tolist(), ["正常A"])

    def test_light_fundamental_filter_should_remove_bad_pe_and_small_cap(self) -> None:
        """轻量基本面过滤应排除负 PE 和过小市值。"""
        data = pd.DataFrame(
            {
                "Symbol": ["000001", "000002", "000003"],
                "Name": ["正常A", "亏损B", "小市值C"],
                "Latest": [10.0, 10.0, 10.0],
                "ChangePct": [3.0, 4.0, 5.0],
                "Amount": [200_000_000, 300_000_000, 400_000_000],
                "TurnoverRate": [2.0, 3.0, 4.0],
                "VolumeRatio": [1.0, 1.0, 1.0],
                "DynamicPE": [30.0, -10.0, 20.0],
                "TotalMarketCap": [10_000_000_000, 10_000_000_000, 500_000_000],
            }
        )

        filtered = apply_light_fundamental_filter(data, StockScreeningConfig())

        self.assertEqual(filtered["Name"].tolist(), ["正常A"])

    def test_rank_candidates_should_sort_by_strength(self) -> None:
        """排序应优先看涨跌幅，再看成交额和换手率。"""
        data = pd.DataFrame(
            {
                "Symbol": ["000001", "000002", "000003"],
                "Name": ["A", "B", "C"],
                "Latest": [10.0, 10.0, 10.0],
                "ChangePct": [3.0, 5.0, 5.0],
                "Amount": [500, 100, 300],
                "TurnoverRate": [2.0, 3.0, 4.0],
                "VolumeRatio": [1.0, 1.0, 1.0],
            }
        )

        ranked = rank_screening_candidates(data, max_candidates=2)

        self.assertEqual(ranked["Name"].tolist(), ["C", "B"])

    def test_fetch_with_retries_should_retry_transient_errors(self) -> None:
        """外部接口临时失败时，应按配置重试。"""
        attempts = {"count": 0}

        def flaky_fetcher():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("远端临时断开")
            return "ok"

        result = fetch_with_retries(
            flaky_fetcher,
            retry_count=2,
            retry_sleep_seconds=0,
        )

        self.assertEqual("ok", result)
        self.assertEqual(3, attempts["count"])

    def test_get_stock_screening_candidates_should_ignore_sector_fetch_error(self) -> None:
        """行业板块接口失败时，不应拖死基础候选池。"""
        raw = pd.DataFrame(
            {
                "代码": ["000001"],
                "名称": ["平安银行"],
                "最新价": [10.0],
                "涨跌幅": [3.5],
                "成交额": [300_000_000],
                "换手率": [2.0],
                "量比": [1.3],
                "市盈率-动态": [12.5],
                "总市值": [300_000_000_000],
            }
        )

        fake_akshare = SimpleNamespace(
            stock_zh_a_spot_em=lambda: raw,
            stock_board_industry_name_em=lambda: (_ for _ in ()).throw(
                RuntimeError("行业板块接口失败")
            ),
        )

        with patch.dict("sys.modules", {"akshare": fake_akshare}):
            candidates = get_stock_screening_candidates(
                StockScreeningConfig(
                    max_candidates=3,
                    fetch_retry_count=0,
                    fetch_retry_sleep_seconds=0,
                )
            )

        self.assertEqual(["000001"], candidates["Symbol"].tolist())
        self.assertIn("Sector", candidates.columns)

    @patch(
        "tradingagents_cn.dataflows.stock_screening.get_industry_board_members",
        side_effect=lambda sector_name: pd.DataFrame(
            {
                "Symbol": ["000001"] if sector_name == "半导体" else ["000002"],
                "Name": ["半导体A"] if sector_name == "半导体" else ["机器人B"],
            }
        ),
    )
    def test_enrich_candidates_with_hot_sectors(self, _mock_members) -> None:
        """候选股应尽量补充所属热门行业板块。"""
        candidates = fake_candidates()
        sectors = pd.DataFrame(
            {
                "Name": ["半导体", "机器人"],
                "ChangePct": [3.2, 2.1],
            }
        )

        enriched = enrich_candidates_with_hot_sectors(candidates, sectors)

        self.assertEqual(enriched.iloc[0]["Sector"], "半导体")
        self.assertEqual(enriched.iloc[1]["Sector"], "机器人")
        self.assertEqual(enriched.iloc[0]["SectorChangePct"], 3.2)

    def test_prompt_should_warn_not_direct_buy(self) -> None:
        """选股 Prompt 必须说明候选名单不是直接买入建议。"""
        prompt = build_stock_screening_agent_prompt(
            question="推荐几只股票",
            materials="候选材料",
        )

        self.assertIn("不是直接买入建议", prompt)
        self.assertIn("最多 5 只", prompt)

    @patch(
        "tradingagents_cn.graph.user_question_pipeline.get_stock_screening_candidates",
        side_effect=lambda config: fake_candidates(),
    )
    def test_run_stock_screening_answer_should_use_llm_output(self, _mock_get) -> None:
        """模型正常返回时，应使用模型输出。"""
        answer = run_stock_screening_answer(
            question="推荐几只股票",
            llm_client=FakeLLMClient(),
        )

        self.assertIn("候选观察名单", answer)
        self.assertNotIn("A 股候选股票池原材料", answer)

    @patch(
        "tradingagents_cn.graph.user_question_pipeline.get_stock_screening_candidates",
        side_effect=lambda config: fake_candidates(),
    )
    def test_run_stock_screening_answer_should_fallback_on_llm_error(self, _mock_get) -> None:
        """模型失败时，应返回候选池原材料。"""
        answer = run_stock_screening_answer(
            question="推荐几只股票",
            llm_client=BrokenLLMClient(),
        )

        self.assertIn("选股模型调用失败", answer)
        self.assertIn("模型超时", answer)
        self.assertIn("A 股候选股票池原材料", answer)

    @patch(
        "tradingagents_cn.graph.user_question_pipeline.run_deep_stock_screening",
        return_value=DeepScreeningResult(
            items=[
                DeepScreeningItem(
                    symbol="000001",
                    name="半导体A",
                    screening_reason="涨跌幅=5.0",
                    action="BUY",
                    rating="Buy",
                    chinese_action="买入",
                    executive_summary="完整单股链路给出积极结论。",
                )
            ],
            errors=[],
        ),
    )
    @patch(
        "tradingagents_cn.graph.user_question_pipeline.get_stock_screening_candidates",
        side_effect=lambda config: fake_candidates(),
    )
    def test_run_stock_screening_answer_should_use_deep_screening_when_enabled(
        self,
        _mock_get,
        mock_deep_screening,
    ) -> None:
        """开启深度筛选时，应把候选股继续送进完整单股链路。"""
        answer = run_stock_screening_answer(
            question="推荐几只股票",
            llm_client=FakeLLMClient(),
            deep_screening=True,
            deep_top_n=1,
        )

        self.assertTrue(mock_deep_screening.called)
        self.assertIn("已完成候选股深度分析", answer)
        self.assertIn("半导体A（000001）", answer)
        self.assertIn("BUY / 买入", answer)
        self.assertNotIn("候选观察名单", answer)

    def test_render_stock_screening_text(self) -> None:
        """候选池文本应包含风险说明。"""
        text = render_stock_screening_text(fake_candidates())

        self.assertIn("不等于最终买入建议", text)
        self.assertIn("半导体A", text)


if __name__ == "__main__":
    unittest.main()
