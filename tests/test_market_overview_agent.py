"""市场概览 Agent 测试。

这些测试不联网，不调用真实大模型。

它们验证：
1. Prompt 是否包含用户问题和市场原材料；
2. 市场概览入口能使用模型输出；
3. 模型失败时能回退到原始市场材料。
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from tradingagents_cn.agents.market_overview_agent import (
    build_market_overview_agent_prompt,
)
from tradingagents_cn.dataflows.market_overview import (
    MarketOverview,
    calculate_market_breadth,
)
from tradingagents_cn.graph.user_question_pipeline import run_market_overview_answer


class FakeLLMClient:
    """测试用成功模型客户端。"""

    def chat(self, messages, tools=None, tool_choice=None, temperature=0.2):
        return {
            "choices": [
                {
                    "message": {
                        "content": "市场状态：偏强。上涨家数占优，半导体领涨。"
                    }
                }
            ]
        }


class BrokenLLMClient:
    """测试用失败模型客户端。"""

    def chat(self, messages, tools=None, tool_choice=None, temperature=0.2):
        raise RuntimeError("模型网络失败")


def fake_market_overview() -> MarketOverview:
    """构造测试用市场概览材料。"""
    return MarketOverview(
        index_snapshot=pd.DataFrame(
            {"Name": ["上证指数"], "Latest": [3000], "ChangePct": [0.5], "Amount": [1000]}
        ),
        market_breadth=calculate_market_breadth(
            pd.DataFrame({"涨跌幅": [1.0, -1.0, 0.0]})
        ),
        sector_snapshot=pd.DataFrame(
            {"Name": ["半导体"], "ChangePct": [2.0], "Amount": [100], "LeadingStock": ["B"]}
        ),
    )


class MarketOverviewAgentTest(unittest.TestCase):
    """测试市场概览 Agent。"""

    def test_prompt_should_include_question_and_materials(self) -> None:
        """Prompt 必须包含用户问题和市场原材料。"""
        prompt = build_market_overview_agent_prompt(
            question="今天股市怎么样？",
            materials="上涨家数：3000",
        )

        self.assertIn("今天股市怎么样？", prompt)
        self.assertIn("上涨家数：3000", prompt)
        self.assertIn("不要推荐具体股票", prompt)

    @patch(
        "tradingagents_cn.graph.user_question_pipeline.get_market_overview",
        side_effect=fake_market_overview,
    )
    def test_run_market_overview_answer_should_use_llm_output(self, _mock_get) -> None:
        """模型正常返回时，应使用模型报告。"""
        answer = run_market_overview_answer(
            question="今天股市怎么样？",
            llm_client=FakeLLMClient(),
        )

        self.assertIn("市场状态：偏强", answer)
        self.assertNotIn("A 股市场概览原材料", answer)

    @patch(
        "tradingagents_cn.graph.user_question_pipeline.get_market_overview",
        side_effect=fake_market_overview,
    )
    def test_run_market_overview_answer_should_fallback_on_llm_error(self, _mock_get) -> None:
        """模型失败时，应返回原始市场材料和错误提示。"""
        answer = run_market_overview_answer(
            question="今天股市怎么样？",
            llm_client=BrokenLLMClient(),
        )

        self.assertIn("市场概览模型调用失败", answer)
        self.assertIn("模型网络失败", answer)
        self.assertIn("A 股市场概览原材料", answer)


if __name__ == "__main__":
    unittest.main()
