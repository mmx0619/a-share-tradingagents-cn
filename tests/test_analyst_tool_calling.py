import unittest

from langchain_core.tools import tool

from tradingagents_cn.graph.analyst_tool_calling import run_analyst_tool_calling_report


@tool
def fake_market_snapshot(symbol: str, trade_date: str) -> str:
    """测试用市场快照工具。"""
    return f"{symbol} 在 {trade_date} 的测试技术面快照。"


class FakeAnalystClient:
    """测试用模型客户端。

    第一次被调用时返回 tool_calls；
    ToolNode 执行工具后，第二次返回最终报告。
    """

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, tool_choice=None, temperature=0.2):
        self.calls += 1
        if self.calls == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_market_1",
                                    "type": "function",
                                    "function": {
                                        "name": "fake_market_snapshot",
                                        "arguments": '{"symbol":"000725","trade_date":"2026-06-18"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        tool_messages = [item for item in messages if item.get("role") == "tool"]
        return {
            "choices": [
                {
                    "message": {
                        "content": f"市场报告：已使用工具材料：{tool_messages[0]['content']}"
                    }
                }
            ]
            }


class RepeatingToolClient:
    """测试用模型：只要还给 tools，就一直想继续调用工具。"""

    def __init__(self):
        self.calls = 0
        self.final_call_without_tools = False

    def chat(self, messages, tools=None, tool_choice=None, temperature=0.2):
        self.calls += 1
        if tools is None:
            self.final_call_without_tools = True
            return {
                "choices": [
                    {
                        "message": {
                            "content": "最终报告：工具调用已达上限，基于已有快照给出分析。"
                        }
                    }
                ]
            }

        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"call_market_{self.calls}",
                                "type": "function",
                                "function": {
                                    "name": "fake_market_snapshot",
                                    "arguments": '{"symbol":"000725","trade_date":"2026-06-18"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }


class AnalystToolCallingTest(unittest.TestCase):
    def test_run_analyst_tool_calling_report_should_execute_toolnode(self):
        """Analyst 执行器应让模型调工具，再基于工具结果输出报告。"""
        client = FakeAnalystClient()

        result = run_analyst_tool_calling_report(
            agent_name="Market Agent",
            system_prompt="你是测试市场分析师。",
            task_prompt="分析 000725。",
            tools=[fake_market_snapshot],
            llm_client=client,
            force_first_tool_name="fake_market_snapshot",
            thread_id="test-analyst-tool-calling",
        )

        self.assertEqual(2, client.calls)
        self.assertEqual(1, result.tool_call_count)
        self.assertIn("市场报告", result.report)
        self.assertTrue(
            any(event["event"] == "assistant_tool_call" for event in result.tool_trace)
        )
        self.assertTrue(any(event["event"] == "tool_result" for event in result.tool_trace))
        self.assertTrue(
            any(event.get("tool_name") == "fake_market_snapshot" for event in result.tool_trace)
        )
        self.assertIn("测试技术面快照", result.report)

    def test_tool_calling_should_stop_after_max_tool_rounds(self):
        """模型反复要求工具时，程序应在上限后强制生成最终报告。"""
        client = RepeatingToolClient()

        result = run_analyst_tool_calling_report(
            agent_name="Market Agent",
            system_prompt="你是测试市场分析师。",
            task_prompt="分析 000725。",
            tools=[fake_market_snapshot],
            llm_client=client,
            force_first_tool_name="fake_market_snapshot",
            thread_id="test-analyst-tool-calling-max-rounds",
            max_tool_rounds=2,
            recursion_limit=20,
        )

        self.assertTrue(client.final_call_without_tools)
        self.assertEqual(3, client.calls)
        self.assertEqual(2, result.tool_call_count)
        self.assertIn("最终报告", result.report)


if __name__ == "__main__":
    unittest.main()
