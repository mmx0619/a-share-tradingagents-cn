import unittest

from tradingagents_cn.graph import ResearchInputConfig, TradingAgentsCNGraph


class FakeLLMClient:
    """测试用假模型客户端。"""

    def chat(self, *args, **kwargs):
        raise AssertionError("本测试不应该真的调用模型")


class TradingAgentsCNGraphTest(unittest.TestCase):
    def test_graph_class_should_sync_selected_analysts_to_config(self):
        """统一封装类应把 selected_analysts 写回 config。"""
        config = ResearchInputConfig(save_full_state=False)
        graph = TradingAgentsCNGraph(
            selected_analysts=("market", "sentiment"),
            config=config,
            llm_client=FakeLLMClient(),
        )

        self.assertEqual(("market", "sentiment"), graph.selected_analysts)
        self.assertEqual(("market", "sentiment"), graph.config.selected_analysts)
        self.assertIsNone(graph.curr_result)


if __name__ == "__main__":
    unittest.main()
