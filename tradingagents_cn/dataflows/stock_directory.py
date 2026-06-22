"""A 股股票名称和代码目录。

这个文件负责把用户说的股票名称转换成 6 位股票代码。

例如：

    京东方A   -> 000725
    大唐发电  -> 601991
    贵州茅台  -> 600519

为什么这个能力很重要？
    你的最终使用方式不是输入代码，
    而是像人一样问：

        帮我看看京东方A能不能买
        大唐发电行情怎么样

    所以程序必须先识别出股票名称和代码，
    然后才能调用完整 TradingAgents 链路。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from tradingagents_cn.dataflows.symbols import normalize_cn_symbol


# 常用股票别名兜底表。
#
# 为什么要有这个表？
#   1. AKShare 股票名称接口有时可能临时失败；
#   2. 用户学习阶段经常用固定几个例子；
#   3. 兜底表能保证这些例子不依赖联网也能被识别。
#
# 后续可以把它扩展成更完整的本地缓存。
LOCAL_STOCK_ALIASES: dict[str, str] = {
    "京东方A": "000725",
    "京东方Ａ": "000725",
    "京东方": "000725",
    "大唐发电": "601991",
    "贵州茅台": "600519",
    "茅台": "600519",
    "神剑股份": "002361",
    "平安银行": "000001",
    "宁德时代": "300750",
    "中际旭创": "300308",
    "江西铜业": "600362",
    "北方稀土": "600111",
    "盛和资源": "600392",
    "新易盛": "300502",
}


@dataclass
class StockSymbolMatch:
    """股票识别结果。

    symbol:
        6 位 A 股代码。

    stock_name:
        股票名称。

    match_source:
        匹配来源。
        常见值：
            code:
                用户直接输入了 6 位代码。
            local_alias:
                命中了本地别名表。
            akshare_directory:
                命中了 AKShare 股票名称表。
    """

    symbol: str
    stock_name: str
    match_source: str


def resolve_stock_from_text(text: str, use_akshare: bool = True) -> StockSymbolMatch | None:
    """从用户自然语言中识别股票。

    识别顺序：

        1. 先看有没有 6 位股票代码。
        2. 再看有没有命中本地别名表。
        3. 如果 use_akshare=True，最后尝试用 AKShare 股票名称表匹配。

    如果没有识别到股票，返回 None。
    """
    matches = resolve_stocks_from_text(text, use_akshare=use_akshare)
    if matches:
        return matches[0]
    return None


def resolve_stocks_from_text(text: str, use_akshare: bool = True) -> list[StockSymbolMatch]:
    """从用户自然语言中识别多只股票。

    为什么需要这个函数？
        用户可能一次问：
            中际旭创，江西铜业，北方稀土，盛和资源，新易盛，怎么样

        旧函数只能返回第一只股票。
        多股问题需要把所有股票识别出来，再逐只进入完整分析链路。

    返回顺序：
        按股票在原问题里出现的顺序返回。
    """
    question = str(text or "").strip()
    if not question:
        return []

    candidates: list[tuple[int, int, StockSymbolMatch]] = []

    def add_candidate(
        start_index: int,
        matched_text: str,
        symbol: str,
        stock_name: str | None,
        match_source: str,
    ) -> None:
        """记录一个候选匹配。

        第二个排序字段用负数长度，
        是为了同一位置同时命中长短名称时优先保留长名称。
        例如“京东方A”优先于“京东方”。
        """
        normalized_symbol = normalize_cn_symbol(symbol)
        candidates.append(
            (
                start_index,
                -len(matched_text),
                StockSymbolMatch(
                    symbol=normalized_symbol,
                    stock_name=stock_name or find_stock_name_by_symbol(normalized_symbol) or normalized_symbol,
                    match_source=match_source,
                ),
            )
        )

    for code_match in re.finditer(r"(?<!\d)(\d{6})(?!\d)", question):
        add_candidate(
            start_index=code_match.start(),
            matched_text=code_match.group(1),
            symbol=code_match.group(1),
            stock_name=find_stock_name_by_symbol(code_match.group(1)),
            match_source="code",
        )

    for alias, symbol in sorted(
        LOCAL_STOCK_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        for alias_match in re.finditer(re.escape(alias), question):
            add_candidate(
                start_index=alias_match.start(),
                matched_text=alias,
                symbol=symbol,
                stock_name=alias,
                match_source="local_alias",
            )

    if not use_akshare:
        return deduplicate_stock_candidates(candidates)

    directory = get_a_share_stock_directory()
    if directory.empty:
        return deduplicate_stock_candidates(candidates)

    # 优先匹配更长的股票名称，避免短名称误匹配。
    directory_candidates = directory.sort_values(
        by="name_length",
        ascending=False,
    )
    for _, row in directory_candidates.iterrows():
        name = str(row["name"])
        if not name:
            continue
        for name_match in re.finditer(re.escape(name), question):
            add_candidate(
                start_index=name_match.start(),
                matched_text=name,
                symbol=str(row["symbol"]),
                stock_name=name,
                match_source="akshare_directory",
            )

    return deduplicate_stock_candidates(candidates)


def deduplicate_stock_candidates(
    candidates: list[tuple[int, int, StockSymbolMatch]],
) -> list[StockSymbolMatch]:
    """股票匹配去重。

    同一个问题里可能同时命中：
        京东方A
        京东方
        000725

    它们都指向同一只股票。
    最终只保留最靠前、名称最长的那一个。
    """
    unique_matches: list[StockSymbolMatch] = []
    seen_symbols: set[str] = set()

    for _, _, stock_match in sorted(candidates, key=lambda item: (item[0], item[1])):
        if stock_match.symbol in seen_symbols:
            continue
        seen_symbols.add(stock_match.symbol)
        unique_matches.append(stock_match)

    return unique_matches


def find_stock_name_by_symbol(symbol: str) -> str | None:
    """根据股票代码查名称。

    先查本地别名表，再查 AKShare 目录。
    """
    normalized_symbol = normalize_cn_symbol(symbol)

    for name, alias_symbol in LOCAL_STOCK_ALIASES.items():
        if normalize_cn_symbol(alias_symbol) == normalized_symbol:
            return name

    directory = get_a_share_stock_directory()
    if directory.empty:
        return None

    matched = directory[directory["symbol"] == normalized_symbol]
    if matched.empty:
        return None

    return str(matched.iloc[0]["name"])


@lru_cache(maxsize=1)
def get_a_share_stock_directory() -> pd.DataFrame:
    """获取 A 股股票代码名称表。

    数据来源：
        AKShare stock_info_a_code_name。

    返回字段统一为：
        symbol
        name
        name_length

    如果 AKShare 调用失败，返回空 DataFrame。
    上层会继续依赖 LOCAL_STOCK_ALIASES 兜底。
    """
    try:
        import akshare as ak

        data = ak.stock_info_a_code_name()
    except Exception:
        return pd.DataFrame(columns=["symbol", "name", "name_length"])

    if data is None or data.empty:
        return pd.DataFrame(columns=["symbol", "name", "name_length"])

    code_column = find_first_existing_column(data, ["code", "代码", "证券代码"])
    name_column = find_first_existing_column(data, ["name", "名称", "证券简称"])
    if code_column is None or name_column is None:
        return pd.DataFrame(columns=["symbol", "name", "name_length"])

    normalized = pd.DataFrame()
    normalized["symbol"] = data[code_column].map(lambda value: normalize_cn_symbol(str(value)))
    normalized["name"] = data[name_column].map(lambda value: str(value).strip())
    normalized = normalized.dropna(subset=["symbol", "name"])
    normalized = normalized[normalized["name"] != ""]
    normalized["name_length"] = normalized["name"].map(len)
    return normalized.drop_duplicates(subset=["symbol", "name"]).reset_index(drop=True)


def find_first_existing_column(data: pd.DataFrame, candidates: list[str]) -> str | None:
    """从候选字段中找到第一个存在的列名。"""
    for column in candidates:
        if column in data.columns:
            return column
    return None
