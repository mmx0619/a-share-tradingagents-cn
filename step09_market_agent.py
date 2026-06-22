"""第 09 步：市场分析师 Agent。

前面几个文件已经完成：

- step03_market_snapshot.py：把行情和指标整理成“市场快照”。
- step04_market_report_prompt.py：把市场快照变成“提示词”。
- step08_llm_client.py：统一调用大模型，支持 DeepSeek / OpenAI / Gemini / Kimi。

当前文件做第九件事：

把“提示词”和“大模型调用”正式组合成一个 Agent。

你可以这样理解：

市场快照文本
  ↓
市场分析师 Prompt
  ↓
大模型
  ↓
市场分析报告

这个文件是后面多智能体系统的第一个正式 Agent。
后续还会有：
- 新闻 Agent
- 情绪 Agent
- 基本面 Agent
- 风控 Agent
- 交易员 Agent
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import step04_market_report_prompt as prompt_mod
import step08_llm_client as llm_mod


@dataclass
class MarketAgentResult:
    """市场分析师 Agent 的返回结果。

    字段说明：
    - symbol：股票代码。
    - provider：模型平台，比如 deepseek、openai、gemini、kimi、mock_llm。
    - model：具体模型名称，比如 deepseek-v4-pro。
    - prompt：发送给大模型的完整提示词。
    - report_text：大模型返回的市场分析报告。

    为什么要把 prompt 也保存下来：
    后面调试 Agent 时，需要知道“模型到底看到了什么”。
    如果只保存报告，不保存 prompt，就很难排查问题。
    """

    symbol: str
    provider: str
    model: str | None
    prompt: str
    report_text: str


def run_market_agent(
    symbol: str,
    market_snapshot_text: str,
    provider: str = "deepseek",
    model: str | None = None,
    temperature: float = 0.2,
) -> MarketAgentResult:
    """运行市场分析师 Agent。

    参数说明：
    - symbol：股票代码，比如 600519。
    - market_snapshot_text：第 03 步生成的市场快照文本。
    - provider：使用哪个模型平台，默认 deepseek。
      也可以传 mock、openai、gemini、kimi。
    - model：具体模型名，不传就使用第 08 步里的默认模型。
    - temperature：模型随机性，投研分析一般建议低一点。

    返回：
    - MarketAgentResult。

    这里的核心流程只有四步：
    1. 用市场快照生成 Prompt。
    2. 根据 provider 创建大模型客户端。
    3. 调用大模型。
    4. 把结果包装成结构化对象。
    """
    prompt = prompt_mod.build_market_prompt_from_text(
        symbol=symbol,
        market_snapshot_text=market_snapshot_text,
    )
    client = llm_mod.create_llm_client(
        provider=provider,
        model=model,
        temperature=temperature,
    )
    response = llm_mod.call_llm(client, prompt)

    return MarketAgentResult(
        symbol=symbol,
        provider=response.provider,
        model=response.model,
        prompt=prompt,
        report_text=response.text,
    )


def render_market_agent_result(result: MarketAgentResult) -> str:
    """把市场分析师 Agent 的结果渲染成容易阅读的文本。"""
    return f"""股票代码：{result.symbol}
模型来源：{result.provider}
模型名称：{result.model}

{result.report_text}
"""


def build_demo_market_snapshot() -> str:
    """构造一段演示市场快照。

    这里先不用真实 AKShare 数据。
    因为当前第 09 步只验证“Agent 能不能调用大模型”。
    后面会再把第 07 步真实 AKShare 流水线和第 09 步 Agent 接起来。
    """
    return """股票代码：600519
交易日期：2024-03-01
收盘价：1680.50
涨跌幅：1.25%
成交量：1234567.00

均线状态：
- 收盘价高于 MA5，短期价格相对强于该均线。
- 收盘价高于 MA10，短期价格相对强于该均线。
- 收盘价低于 MA20，短期价格相对弱于该均线。

量能状态：
- 成交量温和放大，约为 5 日均量的 1.20 倍。

日内收盘位置：
- 收盘接近日内高位，说明尾盘表现偏强。"""


def demo_market_agent() -> None:
    """演示市场分析师 Agent。

    默认使用真实 DeepSeek。

    如果以后想临时切换模型，可以设置环境变量：
    - LLM_PROVIDER=mock
    - LLM_PROVIDER=openai
    - LLM_PROVIDER=gemini
    - LLM_PROVIDER=kimi
    - LLM_PROVIDER=deepseek

    如果想指定具体模型，可以设置：
    - LLM_MODEL=具体模型名
    """
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    model = os.environ.get("LLM_MODEL") or None

    result = run_market_agent(
        symbol="600519",
        market_snapshot_text=build_demo_market_snapshot(),
        provider=provider,
        model=model,
    )
    print(render_market_agent_result(result))


if __name__ == "__main__":
    demo_market_agent()
