"""第 27 步单文件教学版：用最少代码看懂 LangGraph。

为什么要有这个文件？

原来的 step27_langgraph_real_workflow.py 是“工程版”：

- 它复用了前面很多真实模块。
- 它会调用真实行情、新闻、Agent。
- 它更接近最终项目结构。

但是它也有一个问题：

导入文件太多，函数太多，
第一次看 LangGraph 时容易绕晕。

所以这个文件专门做“教学版”：

1. 不导入前面那些 step18、step20、step21、step26。
2. 不联网。
3. 不调用大模型。
4. 只保留 LangGraph 最核心的概念：

   state
   node
   router
   conditional_edges
   compile
   invoke

你可以先看这个文件，
看懂以后再回头看工程版 step27_langgraph_real_workflow.py。
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class SimpleTradingState(TypedDict):
    """LangGraph 里的共享状态。

    你可以把它理解成一个字典。

    整个流程里，每个节点都会读这个字典，也会往这个字典里写东西。

    例如：

    - realtime_node 写入 realtime_quote_text
    - market_node 写入 market_report
    - news_node 写入 news_report
    - summary_node 写入 summary_report
    - risk_node 写入 risk_report
    - trader_node 写入 trader_plan

    路由器就根据这些字段有没有值，判断下一步去哪。
    """

    symbol: str
    realtime_quote_text: str
    market_report: str
    news_report: str
    summary_report: str
    risk_report: str
    trader_plan: str
    step_log: list[str]


def create_initial_state(symbol: str) -> SimpleTradingState:
    """创建初始 state。

    一开始只有股票代码。

    其他字段都是空字符串，
    表示这些节点还没有执行过。
    """
    return {
        "symbol": symbol,
        "realtime_quote_text": "",
        "market_report": "",
        "news_report": "",
        "summary_report": "",
        "risk_report": "",
        "trader_plan": "",
        "step_log": [],
    }


def append_log(state: SimpleTradingState, message: str) -> list[str]:
    """生成新的执行日志列表。

    注意：
    这里不直接 state["step_log"].append(message)。

    原因是：
    在 LangGraph 里，节点函数通常返回“要更新的字段”，
    写成返回新列表更清楚，也更接近 LangGraph 的推荐习惯。
    """
    return [*state["step_log"], message]


def realtime_node(state: SimpleTradingState) -> dict:
    """实时行情节点。

    真实项目里，这一步会去东方财富或新浪拿实时行情。

    这里是教学版，
    所以只写一段模拟文本。

    这个节点执行完以后，
    state["realtime_quote_text"] 就有值了。

    下一次路由器再判断时，
    就不会继续走 realtime_node。
    """
    return {
        "realtime_quote_text": "模拟实时行情：最新价 16.10，涨跌幅 1.07%。",
        "step_log": append_log(state, "执行 realtime_node：写入实时行情"),
    }


def market_node(state: SimpleTradingState) -> dict:
    """市场分析节点。

    真实项目里，这一步会：

    - 获取历史行情
    - 计算均线、成交量等指标
    - 生成市场分析师报告

    教学版只写入一段模拟报告。
    """
    return {
        "market_report": "模拟市场报告：价格低于 MA5，短线偏弱。",
        "step_log": append_log(state, "执行 market_node：写入市场报告"),
    }


def news_node(state: SimpleTradingState) -> dict:
    """新闻分析节点。

    真实项目里，这一步会：

    - 获取新闻
    - 提取龙虎榜、涨停、公告等事件
    - 生成新闻分析报告

    教学版只写入一段模拟报告。
    """
    return {
        "news_report": "模拟新闻报告：近期龙虎榜和高换手较多，短线博弈强。",
        "step_log": append_log(state, "执行 news_node：写入新闻报告"),
    }


def summary_node(state: SimpleTradingState) -> dict:
    """综合汇总节点。

    真实项目里，这一步会把：

    - 实时行情
    - 市场报告
    - 新闻报告

    放进一个 Prompt，
    再让大模型生成综合判断。

    教学版直接拼出一段模拟综合报告。
    """
    summary = (
        "模拟综合报告：实时小幅反弹，但技术面仍偏弱，"
        "新闻面显示短线资金博弈较强。"
    )
    return {
        "summary_report": summary,
        "step_log": append_log(state, "执行 summary_node：写入综合报告"),
    }


def risk_node(state: SimpleTradingState) -> dict:
    """风控节点。

    真实项目里，这一步会读取综合报告，
    判断风险等级、禁止动作、观察条件。

    教学版只写入一段模拟风控报告。
    """
    return {
        "risk_report": "模拟风控报告：维持高风险，不允许主动追涨。",
        "step_log": append_log(state, "执行 risk_node：写入风控报告"),
    }


def trader_node(state: SimpleTradingState) -> dict:
    """交易员节点。

    真实项目里，这一步会读取风控报告，
    然后在风控边界内生成交易预案。

    教学版只写入一段模拟交易预案。
    """
    return {
        "trader_plan": "模拟交易预案：观望，等待放量站回关键均线后再讨论。",
        "step_log": append_log(state, "执行 trader_node：写入交易预案"),
    }


def choose_next_node(state: SimpleTradingState) -> str:
    """路由函数：判断下一步去哪。

    这是 LangGraph 条件边的核心。

    它的规则非常简单：

    - 如果没有实时行情，就去 realtime_node。
    - 如果没有市场报告，就去 market_node。
    - 如果没有新闻报告，就去 news_node。
    - 如果没有综合报告，就去 summary_node。
    - 如果没有风控报告，就去 risk_node。
    - 如果没有交易预案，就去 trader_node。
    - 如果都有了，就返回 done。

    返回值不是给人看的普通文字，
    而是给 LangGraph 用的“节点名”。
    """
    if not state["realtime_quote_text"]:
        return "realtime_node"

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
    """构建 LangGraph 图。

    这一段就是 LangGraph 最核心的写法。

    你可以按顺序看：

    1. 创建图：
       graph = StateGraph(SimpleTradingState)

    2. 注册节点：
       graph.add_node("market_node", market_node)

    3. 注册条件边：
       graph.add_conditional_edges(...)

    4. 编译图：
       graph.compile()
    """
    graph = StateGraph(SimpleTradingState)

    # 注册节点。
    #
    # 左边的字符串是 LangGraph 里的节点名。
    # 右边的是 Python 函数。
    #
    # 例如：
    # "market_node" 对应 market_node 函数。
    graph.add_node("realtime_node", realtime_node)
    graph.add_node("market_node", market_node)
    graph.add_node("news_node", news_node)
    graph.add_node("summary_node", summary_node)
    graph.add_node("risk_node", risk_node)
    graph.add_node("trader_node", trader_node)

    # path_map 告诉 LangGraph：
    #
    # choose_next_node 返回某个字符串时，
    # 应该跳转到哪里。
    #
    # 例如：
    # choose_next_node 返回 "market_node"，
    # LangGraph 就进入 market_node。
    #
    # choose_next_node 返回 "done"，
    # LangGraph 就进入 END，流程结束。
    path_map = {
        "realtime_node": "realtime_node",
        "market_node": "market_node",
        "news_node": "news_node",
        "summary_node": "summary_node",
        "risk_node": "risk_node",
        "trader_node": "trader_node",
        "done": END,
    }

    # START 是 LangGraph 内置的开始节点。
    #
    # 图一开始运行时，
    # 不是固定进入 realtime_node，
    # 而是先调用 choose_next_node 判断。
    graph.add_conditional_edges(
        START,
        choose_next_node,
        path_map=path_map,
    )

    # 每个节点执行完以后，
    # 也不是固定进入下一个节点，
    # 而是再次调用 choose_next_node。
    #
    # 这就是“条件路由”。
    for node_name in [
        "realtime_node",
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


def run_demo(symbol: str = "002361") -> SimpleTradingState:
    """运行单文件 LangGraph 演示。"""
    app = build_graph()
    initial_state = create_initial_state(symbol)

    # invoke 会启动 LangGraph。
    #
    # 从这里开始，
    # LangGraph 会自动根据 choose_next_node 的返回值推进流程。
    final_state = app.invoke(
        initial_state,
        config={"recursion_limit": 20},
    )
    return final_state


def render_final_state(state: SimpleTradingState) -> str:
    """把最终 state 渲染成方便阅读的文本。"""
    step_log = "\n".join(
        f"{index}. {message}"
        for index, message in enumerate(state["step_log"], start=1)
    )
    return f"""股票代码：{state["symbol"]}

======== 执行日志 ========
{step_log}

======== 实时行情 ========
{state["realtime_quote_text"]}

======== 市场报告 ========
{state["market_report"]}

======== 新闻报告 ========
{state["news_report"]}

======== 综合报告 ========
{state["summary_report"]}

======== 风控报告 ========
{state["risk_report"]}

======== 交易预案 ========
{state["trader_plan"]}
"""


if __name__ == "__main__":
    result = run_demo()
    print(render_final_state(result))
