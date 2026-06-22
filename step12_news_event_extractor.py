"""第 12 步：把新闻原文抽取成事件信号。

第 11 步只是把新闻拿回来。
但原始新闻经常很水，比如：

- 龙虎榜数据
- 异动快讯
- 自动生成的行情摘要
- 带免责声明的搬运内容

如果直接把这些新闻全文丢给大模型，价值不高。

当前文件做的是：

新闻原文
  ↓
规则抽取
  ↓
事件类型
  ↓
题材标签
  ↓
风险标签
  ↓
给后续 News Agent 使用的摘要

这一步先不用大模型。
原因是：
先用代码把明显的结构化信息抽出来，
可以减少大模型胡乱发挥，也方便面试时讲清楚数据处理逻辑。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import step11_stock_news as stock_news


@dataclass
class NewsEvent:
    """从一条新闻中抽取出的事件信号。

    字段说明：
    - symbol：股票代码。
    - title：原始新闻标题。
    - publish_time：发布时间。
    - event_types：事件类型，比如 龙虎榜、涨停、跌停、高换手。
    - themes：题材标签，比如 商业航天、人工智能、机器人。
    - numbers：从新闻中抽取出的关键数值，比如换手率、成交额、涨跌幅。
    - risk_tags：风险标签，比如 高换手、短线剧烈波动、龙虎榜博弈。
    - attention_score：关注度分数，越高说明越值得后续 Agent 重点阅读。
    - summary：把抽取结果整理成一句话。
    """

    symbol: str
    title: str
    publish_time: str
    event_types: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    numbers: dict[str, str] = field(default_factory=dict)
    risk_tags: list[str] = field(default_factory=list)
    attention_score: int = 0
    summary: str = ""


THEME_KEYWORDS: dict[str, list[str]] = {
    "商业航天": ["商业航天", "航天", "卫星", "火箭"],
    "人工智能": ["人工智能", "AI", "大模型", "算力"],
    "机器人": ["机器人", "减速器", "人形机器人"],
    "低空经济": ["低空经济", "无人机", "eVTOL"],
    "新能源": ["新能源", "锂电", "光伏", "储能"],
    "半导体": ["半导体", "芯片", "集成电路"],
    "白酒": ["白酒", "茅台", "酒"],
}


def unique_keep_order(values: list[str]) -> list[str]:
    """去重但保留原始顺序。"""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def combined_news_text(item: stock_news.StockNewsItem) -> str:
    """把标题和内容拼在一起，方便统一做关键词匹配。"""
    return clean_news_text(f"{item.title}\n{item.content}")


def clean_news_text(text: str) -> str:
    """清理会干扰事件抽取的固定模板文字。

    有些东方财富新闻会带类似：
    “免责声明：本文基于AI生产，仅供参考……”

    这句话里的 “AI” 不是股票题材，
    如果不清理，会误判为“人工智能”题材。
    """
    text = re.sub(r"免责声明：本文基于AI生产.*", "", text)
    text = re.sub(r"免责声明：.*", "", text)
    return text.strip()


def extract_event_types(text: str) -> list[str]:
    """从新闻文本中识别事件类型。"""
    event_types: list[str] = []

    if "龙虎榜" in text:
        event_types.append("龙虎榜")
    if "涨停" in text:
        event_types.append("涨停")
    if "跌停" in text or "日跌幅偏离值" in text:
        event_types.append("跌停/大跌")
    if "日换手率达到20%" in text or "换手率" in text:
        event_types.append("高换手")
    if "异动" in text:
        event_types.append("盘中异动")
    if "公告" in text:
        event_types.append("公司公告")
    if "董事会秘书" in text or "董秘" in text or "董事长" in text:
        event_types.append("高管变动/治理事件")

    return unique_keep_order(event_types)


def extract_themes(text: str) -> list[str]:
    """从新闻文本中识别题材标签。"""
    themes: list[str] = []
    for theme, keywords in THEME_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            themes.append(theme)
    return unique_keep_order(themes)


def extract_numbers(text: str) -> dict[str, str]:
    """从新闻文本中抽取常见交易数值。

    这里不是做复杂自然语言理解，只抓最常见、最有用的几类数字：
    - 涨跌幅
    - 换手率
    - 成交额
    - 收盘价
    - 偏离值
    """
    patterns = {
        "涨跌幅": r"涨跌幅(-?\d+(?:\.\d+)?%)",
        "换手率": r"换手率(\d+(?:\.\d+)?%)",
        "成交额": r"成交额(\d+(?:\.\d+)?亿)",
        "收盘价": r"收报(\d+(?:\.\d+)?)元",
        "偏离值": r"偏离\s*值?(-?\d+(?:\.\d+)?%)",
    }

    numbers: dict[str, str] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            numbers[name] = match.group(1)
    return numbers


def build_risk_tags(
    event_types: list[str],
    numbers: dict[str, str],
    themes: list[str],
) -> list[str]:
    """根据事件类型和数字生成风险标签。"""
    risk_tags: list[str] = []

    if "龙虎榜" in event_types:
        risk_tags.append("龙虎榜资金博弈")
    if "高换手" in event_types:
        risk_tags.append("高换手")
    if "涨停" in event_types or "跌停/大跌" in event_types:
        risk_tags.append("短线剧烈波动")
    if themes:
        risk_tags.append("题材驱动")

    turnover = numbers.get("换手率")
    if turnover:
        turnover_value = float(turnover.rstrip("%"))
        if turnover_value >= 30:
            risk_tags.append("换手极高")
        elif turnover_value >= 20:
            risk_tags.append("换手偏高")

    amount = numbers.get("成交额")
    if amount:
        amount_value = float(amount.rstrip("亿"))
        if amount_value >= 50:
            risk_tags.append("成交额巨大")

    return unique_keep_order(risk_tags)


def calculate_attention_score(
    event_types: list[str],
    risk_tags: list[str],
    themes: list[str],
) -> int:
    """计算新闻关注度分数。

    这个分数不是投资建议，只是告诉后续 Agent：
    哪些新闻更值得优先阅读。
    """
    score = 0

    score += len(event_types) * 2
    score += len(risk_tags) * 2
    score += len(themes)

    if "龙虎榜" in event_types:
        score += 3
    if "短线剧烈波动" in risk_tags:
        score += 3
    if "换手极高" in risk_tags:
        score += 3

    return score


def build_event_summary(event: NewsEvent) -> str:
    """把事件抽取结果整理成一句话摘要。"""
    event_text = "、".join(event.event_types) if event.event_types else "普通新闻"
    theme_text = "、".join(event.themes) if event.themes else "暂无明确题材"
    risk_text = "、".join(event.risk_tags) if event.risk_tags else "暂无明显风险标签"

    number_parts = [f"{key} {value}" for key, value in event.numbers.items()]
    number_text = "，".join(number_parts) if number_parts else "未抽取到关键交易数值"

    return (
        f"事件类型：{event_text}；"
        f"题材：{theme_text}；"
        f"关键数值：{number_text}；"
        f"风险标签：{risk_text}。"
    )


def extract_news_event(item: stock_news.StockNewsItem) -> NewsEvent:
    """从单条新闻中抽取事件信号。"""
    text = combined_news_text(item)
    event_types = extract_event_types(text)
    themes = extract_themes(text)
    numbers = extract_numbers(text)
    risk_tags = build_risk_tags(event_types, numbers, themes)
    attention_score = calculate_attention_score(event_types, risk_tags, themes)

    event = NewsEvent(
        symbol=item.symbol,
        title=item.title,
        publish_time=item.publish_time,
        event_types=event_types,
        themes=themes,
        numbers=numbers,
        risk_tags=risk_tags,
        attention_score=attention_score,
    )
    event.summary = build_event_summary(event)
    return event


def extract_news_events(news_items: list[stock_news.StockNewsItem]) -> list[NewsEvent]:
    """从多条新闻中抽取事件信号，并按关注度从高到低排序。"""
    events = [extract_news_event(item) for item in news_items]
    return sorted(events, key=lambda event: event.attention_score, reverse=True)


def render_news_events_text(events: list[NewsEvent]) -> str:
    """把事件信号渲染成适合人和后续 Agent 阅读的文本。"""
    if not events:
        return "暂无可用新闻事件。"

    blocks: list[str] = []
    for index, event in enumerate(events, start=1):
        blocks.append(
            f"""事件 {index}
股票代码：{event.symbol}
标题：{event.title}
时间：{event.publish_time}
关注度：{event.attention_score}
事件类型：{", ".join(event.event_types) if event.event_types else "无"}
题材标签：{", ".join(event.themes) if event.themes else "无"}
风险标签：{", ".join(event.risk_tags) if event.risk_tags else "无"}
关键数值：{event.numbers if event.numbers else "无"}
摘要：{event.summary}"""
        )
    return "\n\n".join(blocks)


def demo_news_event_extractor() -> None:
    """演示从真实新闻中抽取事件信号。"""
    news_items = stock_news.get_stock_news("002361", max_items=5)
    events = extract_news_events(news_items)
    print(render_news_events_text(events))


if __name__ == "__main__":
    demo_news_event_extractor()
