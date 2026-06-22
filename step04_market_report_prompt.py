"""第 04 步：把市场快照转换成“市场分析师”的提示词。

前面三个文件已经完成：
- step01_akshare_cn.py：获取 A 股日线行情。
- step02_technical_indicators.py：计算基础技术指标。
- step03_market_snapshot.py：把指标整理成市场快照。

当前文件做第四件事：
把市场快照文本包装成一个清晰的 Prompt，后面可以交给大模型。

注意：
这个文件不调用大模型，也不依赖 OpenAI / DeepSeek / Qwen。
它只负责生成提示词。

为什么要单独做 Prompt 文件：
1. Prompt 是 Agent 的“岗位说明书”。
2. 把 Prompt 单独拆出来，方便后面修改和面试讲解。
3. 后续接入大模型时，只需要把这里生成的文本发给模型。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MarketPromptInput:
    """市场分析师 Prompt 的输入数据。

    目前只需要两个字段：
    - symbol：股票代码。
    - market_snapshot_text：第 03 步生成的市场快照中文文本。

    后续可以继续扩展：
    - stock_name：股票名称。
    - industry：所属行业。
    - index_context：大盘环境。
    - sector_context：板块环境。
    """

    symbol: str
    market_snapshot_text: str


def build_market_analyst_prompt(prompt_input: MarketPromptInput) -> str:
    """生成市场分析师 Prompt。

    这个 Prompt 的目标：
    让大模型扮演 A 股市场技术分析师，
    根据我们前面用代码整理出来的市场快照，
    输出一份技术面分析报告。

    注意：
    这里不要求模型给最终买卖建议。
    原因是：
    技术面只是多智能体系统中的一个 Agent。
    最终买卖建议应该等新闻、情绪、基本面、风控都分析完以后再决定。
    """
    return f"""你是一名 A 股市场技术分析师。

你的任务是根据给定的市场快照，写一份技术面分析报告。

请遵守以下要求：

1. 只基于我提供的市场快照分析，不要编造不存在的数据。
2. 如果某个字段显示“数据不足”，请明确说明该指标暂时无法判断。
3. 不要直接给出最终买入、卖出建议。
4. 可以判断短期趋势、均线状态、量能状态和日内强弱。
5. 需要区分“事实”和“推断”。
6. 输出要适合后续交易员 Agent 阅读。

分析重点：

- 收盘价相对 MA5、MA10、MA20 的位置。
- 当日涨跌幅是否体现短期强弱。
- 成交量相对 5 日均量是放大、缩小还是正常。
- 收盘价在当日 K 线中的位置，判断尾盘强弱。
- 如果多个信号互相矛盾，需要指出分歧。
- 如果多个信号方向一致，需要指出共振。

输出格式：

请按照以下结构输出：

## 技术面结论
用 2-4 句话总结当前技术面状态。

## 关键证据
用项目符号列出支持结论的关键数据。

## 风险与分歧
说明有哪些信号不确定、数据不足或互相矛盾。

## 给后续 Agent 的提示
告诉后续新闻、情绪、基本面或风控 Agent 应该重点关注什么。

股票代码：
{prompt_input.symbol}

市场快照：
{prompt_input.market_snapshot_text}
"""


def build_market_prompt_from_text(symbol: str, market_snapshot_text: str) -> str:
    """用普通参数快速生成市场分析师 Prompt。

    这个函数是一个便利入口。
    如果不想手动创建 MarketPromptInput，可以直接调用它。
    """
    prompt_input = MarketPromptInput(
        symbol=symbol,
        market_snapshot_text=market_snapshot_text,
    )
    return build_market_analyst_prompt(prompt_input)


if __name__ == "__main__":
    # 这里放一段假的市场快照文本，方便直接运行当前文件查看 Prompt 长什么样。
    # 注意：这里不联网，也不调用大模型。
    demo_snapshot = """股票代码：600519
交易日期：2024-01-10
收盘价：1680.50
涨跌幅：1.25%
成交量：1234567.00

均线状态：
- 收盘价高于 MA5，短期价格相对强于该均线。
- 收盘价高于 MA10，短期价格相对强于该均线。
- MA20 数据不足，暂时无法判断。

量能状态：
- 成交量温和放大，约为 5 日均量的 1.20 倍。

日内收盘位置：
- 收盘接近日内高位，说明尾盘表现偏强。"""

    prompt = build_market_prompt_from_text("600519", demo_snapshot)
    print(prompt)
