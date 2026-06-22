import unittest

from langchain_core.tools import tool

from tradingagents_cn.graph.langgraph_toolnode_workflow import (
    run_langgraph_toolnode_workflow,
)


@tool
def fake_quote(symbol: str) -> str:
    """测试用行情工具。"""
    return f"{symbol} 测试行情：上涨 1%。"


class FakeToolCallingClient:
    """测试用模型：第一次要求调用工具，第二次输出最终回答。"""

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
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "fake_quote",
                                        "arguments": '{"symbol": "000725"}',
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
                        "content": f"最终回答：已经读取工具结果：{tool_messages[0]['content']}"
                    }
                }
            ]
        }


class LangGraphToolNodeWorkflowTest(unittest.TestCase):
    def test_workflow_should_execute_toolnode_and_return_final_answer(self):
        """StateGraph 应通过 ToolNode 执行工具，再回到模型生成回答。"""
        client = FakeToolCallingClient()

        result = run_langgraph_toolnode_workflow(
            question="000725 现在行情怎么样？",
            llm_client=client,
            tools=[fake_quote],
            thread_id="test-thread",
        )

        self.assertEqual(2, client.calls)
        self.assertEqual("test-thread", result.thread_id)
        self.assertIn("最终回答", result.final_answer)
        self.assertIn("000725 测试行情", result.final_answer)


if __name__ == "__main__":
    unittest.main()
