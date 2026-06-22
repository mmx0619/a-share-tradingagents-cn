"""第 28 步：用单文件看懂 LangGraph 里的 Agent 和 Prompt 流转。

第 27 步单文件教学版讲的是：

LangGraph 如何根据 state 路由到不同节点。

这一文件继续讲下一个问题：

Agent 到底做了什么？

你之前的理解是对的：

上一个 Agent 的输出
  ↓
放进下一个 Agent 的 Prompt
  ↓
再交给大模型
  ↓
得到下一个 Agent 的输出

本文件就专门演示这件事。

为了学习清楚：

1. 不联网。
2. 不调用真实 DeepSeek。
3. 不导入前面的复杂模块。
4. 只使用 LangGraph。
5. 用 mock_llm(prompt) 模拟大模型。

你要重点看：

- 每个节点如何构造 Prompt。
- Prompt 里如何包含上一步结果。
- mock_llm 如何根据 Prompt 返回文本。
- 返回文本如何写入 state。
- 路由器如何根据 state 判断下一步。
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class PromptFlowState(TypedDict):
    """LangGraph 共享状态。

    这个 state 比第 27 步多了一些 prompt 字段。

    为什么要保存 prompt？

    因为你正在学习 Agent 的内部过程。
    如果只保存报告，不保存 Prompt，
    你会看不到“大模型到底收到了什么输入”。

    字段分两类：

    第一类：Agent 输出
    - market_report
    - news_report
    - summary_report
    - risk_report
    - trader_plan

    第二类：发送给模型的 Prompt
    - market_prompt
    - news_prompt
    - summary_prompt
    - risk_prompt
    - trader_prompt
    """

    symbol: str
    realtime_quote_text: str
    market_prompt: str
    market_report: str
    news_prompt: str
    news_report: str
    summary_prompt: str
    summary_report: str
    risk_prompt: str
    risk_report: str
    trader_prompt: str
    trader_plan: str
    step_log: list[str]


def create_initial_state(symbol: str) -> PromptFlowState:
    """创建初始状态。

    教学版里，实时行情直接放一段模拟文本。

    为什么不做 realtime_node？

    因为第 28 步的重点不是行情采集，
    而是 Agent 和 Prompt 的流转。
    """
    return {
        "symbol": symbol,
        "realtime_quote_text": "模拟实时行情：最新价 16.10，涨跌幅 1.07%。",
        "market_prompt": "",
        "market_report": "",
        "news_prompt": "",
        "news_report": "",
        "summary_prompt": "",
        "summary_report": "",
        "risk_prompt": "",
        "risk_report": "",
        "trader_prompt": "",
        "trader_plan": "",
        "step_log": [],
    }


def append_log(state: PromptFlowState, message: str) -> list[str]:
    """返回新的日志列表。"""
    return [*state["step_log"], message]


def mock_llm(prompt: str) -> str:
    """模拟大模型。

    真实系统里，这里会调用：

    - DeepSeek
    - OpenAI
    - Gemini
    - Kimi

    但教学版不调真实 API。

    这里根据 Prompt 开头的角色关键词，
    返回不同的模拟报告。

    你可以把它理解成：

    输入：一大段 Prompt 文本
    输出：一段模型生成的文本

    注意：
    不能简单写成：

        if "市场分析师" in prompt:

    因为后面的新闻 Prompt、综合 Prompt 里面，
    也可能包含“上一步市场分析师报告”这几个字。

    如果只用 in 判断，
    新闻 Agent 也会被误判成市场分析师。

    所以这里使用 startswith()，
    只看 Prompt 一开始声明的角色。
    """
    if prompt.startswith("你是 A 股多智能体系统里的市场分析师"):
        return (
            "市场分析师报告：实时价格有小幅反弹，"
            "但价格仍低于关键均线，短线技术面偏谨慎。"
        )

    if prompt.startswith("你是 A 股多智能体系统里的新闻分析师"):
        return (
            "新闻分析师报告：近期龙虎榜、高换手等信号较多，"
            "说明短线资金博弈强，消息面风险不低。"
        )

    if prompt.startswith("你是 A 股多智能体系统里的综合研究经理"):
        return (
            "综合研究报告：技术面偏弱，新闻面显示短线博弈强。"
            "实时小幅反弹暂时不能证明趋势反转。"
        )

    if prompt.startswith("你是 A 股多智能体系统里的风控 Agent"):
        return (
            "风控报告：维持高风险等级。"
            "禁止主动追涨，允许继续观察量能和关键均线修复。"
        )

    if prompt.startswith("你是 A 股多智能体系统里的交易员 Agent"):
        return (
            "交易预案：当前选择观望。"
            "只有放量站回关键均线并且风险等级下调后，才重新讨论交易方案。"
        )

    return "模拟模型输出：未识别角色。"


def market_node(state: PromptFlowState) -> dict:
    """市场分析师 Agent 节点。

    这个节点做三件事：

    1. 构造市场分析师 Prompt。
    2. 把 Prompt 交给 mock_llm。
    3. 把模型输出写入 state["market_report"]。

    注意：
    这个节点主要读取实时行情。
    """
    prompt = f"""你是 A 股多智能体系统里的市场分析师。

请基于实时行情分析技术面，不要给最终买卖建议。

股票代码：
{state["symbol"]}

实时行情：
{state["realtime_quote_text"]}
"""
    report = mock_llm(prompt)
    return {
        "market_prompt": prompt,
        "market_report": report,
        "step_log": append_log(state, "market_node：生成市场 Prompt，并得到市场报告"),
    }


def news_node(state: PromptFlowState) -> dict:
    """新闻分析师 Agent 节点。

    这个节点演示：

    新闻 Agent 不一定只看新闻，
    它也可以参考前面的市场报告。

    也就是说：

    market_report
      ↓
    放进 news_prompt
      ↓
    mock_llm
      ↓
    news_report
    """
    prompt = f"""你是 A 股多智能体系统里的新闻分析师。

请结合市场分析师报告，判断消息面是否放大风险。

股票代码：
{state["symbol"]}

上一步市场分析师报告：
{state["market_report"]}

模拟新闻材料：
近期出现龙虎榜、高换手、短线剧烈波动等信息。
"""
    report = mock_llm(prompt)
    return {
        "news_prompt": prompt,
        "news_report": report,
        "step_log": append_log(state, "news_node：把市场报告放入新闻 Prompt，并得到新闻报告"),
    }


def summary_node(state: PromptFlowState) -> dict:
    """综合研究经理 Agent 节点。

    这个节点最能体现“上游 Agent 输出进入下游 Prompt”。

    它读取：

    - realtime_quote_text
    - market_report
    - news_report

    然后把这些内容全部放进综合 Prompt。
    """
    prompt = f"""你是 A 股多智能体系统里的综合研究经理。

你的任务不是直接交易，
而是综合市场分析师和新闻分析师的结论。

股票代码：
{state["symbol"]}

实时行情：
{state["realtime_quote_text"]}

市场分析师报告：
{state["market_report"]}

新闻分析师报告：
{state["news_report"]}
"""
    report = mock_llm(prompt)
    return {
        "summary_prompt": prompt,
        "summary_report": report,
        "step_log": append_log(state, "summary_node：合并市场报告和新闻报告，得到综合报告"),
    }


def risk_node(state: PromptFlowState) -> dict:
    """风控 Agent 节点。

    风控 Agent 读取综合报告，
    不是为了寻找机会，
    而是为了确定风险边界。
    """
    prompt = f"""你是 A 股多智能体系统里的风控 Agent。

请根据综合研究报告判断风险等级和禁止动作。

股票代码：
{state["symbol"]}

综合研究报告：
{state["summary_report"]}
"""
    report = mock_llm(prompt)
    return {
        "risk_prompt": prompt,
        "risk_report": report,
        "step_log": append_log(state, "risk_node：把综合报告放入风控 Prompt，并得到风控报告"),
    }


def trader_node(state: PromptFlowState) -> dict:
    """交易员 Agent 节点。

    交易员 Agent 读取风控报告。

    关键点：

    如果风控报告说禁止追涨，
    交易员就不能自己突破风控限制。
    """
    prompt = f"""你是 A 股多智能体系统里的交易员 Agent。

请严格服从风控报告，只输出条件式交易预案。

股票代码：
{state["symbol"]}

风控报告：
{state["risk_report"]}
"""
    plan = mock_llm(prompt)
    return {
        "trader_prompt": prompt,
        "trader_plan": plan,
        "step_log": append_log(state, "trader_node：把风控报告放入交易员 Prompt，并得到交易预案"),
    }


def choose_next_node(state: PromptFlowState) -> str:
    """根据 state 判断下一步去哪。"""
    if not state["market_report"]:
        return "market_node"

    if not state["news_report"]:
        return "news_node"

    if not state["summary_report"]:
        return "summary_node"

    if not state["risk_report"]:
        return "risk_node"

    if not state["trader_plan"]:
        return "trader_node"

    return "done"


def build_graph():
    """构建 LangGraph。

    这里和第 27 步类似：

    - add_node 注册 Agent 节点。
    - add_conditional_edges 注册条件路由。
    - compile 编译图。
    """
    graph = StateGraph(PromptFlowState)

    graph.add_node("market_node", market_node)
    graph.add_node("news_node", news_node)
    graph.add_node("summary_node", summary_node)
    graph.add_node("risk_node", risk_node)
    graph.add_node("trader_node", trader_node)

    path_map = {
        "market_node": "market_node",
        "news_node": "news_node",
        "summary_node": "summary_node",
        "risk_node": "risk_node",
        "trader_node": "trader_node",
        "done": END,
    }

    graph.add_conditional_edges(
        START,
        choose_next_node,
        path_map=path_map,
    )

    for node_name in [
        "market_node",
        "news_node",
        "summary_node",
        "risk_node",
        "trader_node",
    ]:
        graph.add_conditional_edges(
            node_name,
            choose_next_node,
            path_map=path_map,
        )

    return graph.compile()


def run_demo(symbol: str = "002361") -> PromptFlowState:
    """运行 Prompt 流转演示。"""
    app = build_graph()
    initial_state = create_initial_state(symbol)
    return app.invoke(
        initial_state,
        config={"recursion_limit": 20},
    )


def render_prompt_flow_report(state: PromptFlowState) -> str:
    """渲染最终结果。

    这里不仅输出最终报告，
    还输出每个 Agent 的 Prompt。

    这样你可以直接看到：

    上一个 Agent 的输出，是如何进入下一个 Prompt 的。
    """
    step_log = "\n".join(
        f"{index}. {message}"
        for index, message in enumerate(state["step_log"], start=1)
    )
    return f"""股票代码：{state["symbol"]}

======== 执行日志 ========
{step_log}

======== 市场分析师 Prompt ========
{state["market_prompt"]}

======== 市场分析师输出 ========
{state["market_report"]}

======== 新闻分析师 Prompt ========
{state["news_prompt"]}

======== 新闻分析师输出 ========
{state["news_report"]}

======== 综合研究经理 Prompt ========
{state["summary_prompt"]}

======== 综合研究经理输出 ========
{state["summary_report"]}

======== 风控 Agent Prompt ========
{state["risk_prompt"]}

======== 风控 Agent 输出 ========
{state["risk_report"]}

======== 交易员 Agent Prompt ========
{state["trader_prompt"]}

======== 交易员 Agent 输出 ========
{state["trader_plan"]}
"""


if __name__ == "__main__":
    result = run_demo()
    
    print(render_prompt_flow_report(result))
