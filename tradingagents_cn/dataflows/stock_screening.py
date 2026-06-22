"""A 股选股/筛股数据层。

这个文件服务于用户这类问题：

    帮我筛几只短线机会
    今天有什么股票值得关注
    推荐几个强势股看看

当前版本先做“行情快照筛选”：

1. 从全市场实时行情里读取股票；
2. 清洗字段；
3. 过滤 ST、退市、价格异常、成交额过低的股票；
4. 用轻量估值/市值字段过滤明显异常候选；
5. 按涨跌幅、成交额、换手率等指标排序；
6. 尽量补充所属热门行业板块；
7. 输出候选股票池材料。

注意：
    这不是最终买入建议。
    它只是给 Stock Screening Agent 的候选原材料。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from tradingagents_cn.dataflows.realtime_quote import normalize_quote_symbol, to_float
from tradingagents_cn.dataflows.market_overview import normalize_sector_snapshot


@dataclass
class StockScreeningConfig:
    """选股数据层配置。"""

    max_candidates: int = 20
    min_amount: float = 100_000_000
    min_price: float = 2.0
    exclude_st: bool = True
    sector_top_n: int = 10
    enable_light_fundamental_filter: bool = True
    max_dynamic_pe: float = 300.0
    min_total_market_cap: float = 2_000_000_000
    fetch_retry_count: int = 2
    fetch_retry_sleep_seconds: float = 1.0


def get_stock_screening_candidates(
    config: StockScreeningConfig | None = None,
) -> pd.DataFrame:
    """获取 A 股候选股票池。"""
    import akshare as ak

    actual_config = config or StockScreeningConfig()
    raw = fetch_with_retries(
        ak.stock_zh_a_spot_em,
        retry_count=actual_config.fetch_retry_count,
        retry_sleep_seconds=actual_config.fetch_retry_sleep_seconds,
    )

    try:
        sector_raw = fetch_with_retries(
            ak.stock_board_industry_name_em,
            retry_count=actual_config.fetch_retry_count,
            retry_sleep_seconds=actual_config.fetch_retry_sleep_seconds,
        )
    except Exception:
        # 行业板块只是候选股的补充信息。
        # 如果这个接口临时失败，不应该让整轮自动交易直接没有候选。
        sector_raw = pd.DataFrame()

    normalized = normalize_spot_frame(raw)
    filtered = filter_screening_candidates(normalized, actual_config)
    ranked = rank_screening_candidates(
        filtered,
        max_candidates=actual_config.max_candidates,
    )
    sector_snapshot = normalize_sector_snapshot(
        sector_raw,
        top_n=actual_config.sector_top_n,
    )
    return enrich_candidates_with_hot_sectors(ranked, sector_snapshot)


def fetch_with_retries(
    fetcher: Callable[[], Any],
    retry_count: int = 2,
    retry_sleep_seconds: float = 1.0,
) -> Any:
    """调用外部数据接口，遇到临时网络错误时重试。

    AKShare 背后很多接口来自公开网页。
    公开网页偶尔会断开连接、限流或返回空响应。
    对这种临时问题，程序重试几次比直接失败更适合无人值守运行。
    """
    max_attempts = 1 + max(0, int(retry_count))
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return fetcher()
        except Exception as error:
            last_error = error
            if attempt >= max_attempts - 1:
                break
            if retry_sleep_seconds > 0:
                time.sleep(float(retry_sleep_seconds))

    if last_error is not None:
        raise last_error
    raise RuntimeError("外部数据接口调用失败。")


def normalize_spot_frame(data: pd.DataFrame) -> pd.DataFrame:
    """把东方财富全市场快照整理成稳定字段。"""
    if data is None or data.empty:
        return pd.DataFrame(
            columns=[
                "Symbol",
                "Name",
                "Latest",
                "ChangePct",
                "Amount",
                "TurnoverRate",
                "VolumeRatio",
                "DynamicPE",
                "PB",
                "TotalMarketCap",
                "FloatMarketCap",
            ]
        )

    code_column = find_first_existing_column(data, ["代码", "code", "symbol"])
    name_column = find_first_existing_column(data, ["名称", "name"])
    latest_column = find_first_existing_column(data, ["最新价", "price"])
    change_pct_column = find_first_existing_column(data, ["涨跌幅", "change_pct"])
    amount_column = find_first_existing_column(data, ["成交额", "amount"])
    turnover_column = find_first_existing_column(data, ["换手率", "turnover"])
    volume_ratio_column = find_first_existing_column(data, ["量比", "volume_ratio"])
    dynamic_pe_column = find_first_existing_column(
        data,
        ["市盈率-动态", "动态市盈率", "市盈率", "pe", "DynamicPE"],
    )
    pb_column = find_first_existing_column(data, ["市净率", "pb", "PB"])
    total_market_cap_column = find_first_existing_column(
        data,
        ["总市值", "total_market_cap", "TotalMarketCap"],
    )
    float_market_cap_column = find_first_existing_column(
        data,
        ["流通市值", "float_market_cap", "FloatMarketCap"],
    )

    if code_column is None or name_column is None:
        return pd.DataFrame(
            columns=[
                "Symbol",
                "Name",
                "Latest",
                "ChangePct",
                "Amount",
                "TurnoverRate",
                "VolumeRatio",
                "DynamicPE",
                "PB",
                "TotalMarketCap",
                "FloatMarketCap",
            ]
        )

    frame = pd.DataFrame()
    frame["Symbol"] = data[code_column].map(safe_normalize_symbol)
    frame["Name"] = data[name_column].map(lambda value: str(value).strip())
    frame["Latest"] = data[latest_column].map(to_float) if latest_column else None
    frame["ChangePct"] = data[change_pct_column].map(to_float) if change_pct_column else None
    frame["Amount"] = data[amount_column].map(to_float) if amount_column else None
    frame["TurnoverRate"] = data[turnover_column].map(to_float) if turnover_column else None
    frame["VolumeRatio"] = data[volume_ratio_column].map(to_float) if volume_ratio_column else None
    frame["DynamicPE"] = data[dynamic_pe_column].map(to_float) if dynamic_pe_column else None
    frame["PB"] = data[pb_column].map(to_float) if pb_column else None
    frame["TotalMarketCap"] = (
        data[total_market_cap_column].map(to_float)
        if total_market_cap_column
        else None
    )
    frame["FloatMarketCap"] = (
        data[float_market_cap_column].map(to_float)
        if float_market_cap_column
        else None
    )
    return frame.dropna(subset=["Symbol"]).reset_index(drop=True)


def filter_screening_candidates(
    data: pd.DataFrame,
    config: StockScreeningConfig,
) -> pd.DataFrame:
    """过滤明显不适合作为候选的股票。"""
    if data is None or data.empty:
        return pd.DataFrame()

    frame = data.copy()
    frame = frame.dropna(subset=["Latest", "ChangePct", "Amount"])
    frame = frame[frame["Latest"] >= config.min_price]
    frame = frame[frame["Amount"] >= config.min_amount]

    if config.exclude_st:
        frame = frame[~frame["Name"].str.contains("ST|退", case=False, regex=True, na=False)]

    if config.enable_light_fundamental_filter:
        frame = apply_light_fundamental_filter(frame, config)

    return frame.reset_index(drop=True)


def apply_light_fundamental_filter(
    data: pd.DataFrame,
    config: StockScreeningConfig,
) -> pd.DataFrame:
    """用轻量估值/市值字段过滤明显异常候选。

    注意：
        这不是完整基本面分析。
        它只利用全市场快照里常见的轻量字段：
            DynamicPE
            TotalMarketCap

    如果字段缺失或为空，不强制过滤。
    这样可以避免 AKShare 字段变化导致候选池被误清空。
    """
    if data is None or data.empty:
        return pd.DataFrame()

    frame = data.copy()

    if "DynamicPE" in frame.columns:
        pe = pd.to_numeric(frame["DynamicPE"], errors="coerce")
        pe_mask = pe.isna() | ((pe > 0) & (pe <= config.max_dynamic_pe))
        frame = frame[pe_mask]

    if "TotalMarketCap" in frame.columns:
        market_cap = pd.to_numeric(frame["TotalMarketCap"], errors="coerce")
        cap_mask = market_cap.isna() | (market_cap >= config.min_total_market_cap)
        frame = frame[cap_mask]

    return frame


def rank_screening_candidates(data: pd.DataFrame, max_candidates: int = 20) -> pd.DataFrame:
    """对候选股票排序。

    当前排序偏短线市场强度：
        1. 涨跌幅高；
        2. 成交额高；
        3. 换手率高。
    """
    if data is None or data.empty:
        return pd.DataFrame()

    frame = data.copy()
    frame["TurnoverRate"] = frame["TurnoverRate"].fillna(0)
    frame = frame.sort_values(
        by=["ChangePct", "Amount", "TurnoverRate"],
        ascending=[False, False, False],
    )
    return frame.head(max(1, int(max_candidates))).reset_index(drop=True)


def enrich_candidates_with_hot_sectors(
    candidates: pd.DataFrame,
    sector_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """给候选股补充热门行业板块信息。

    AKShare 的行业板块成分需要逐个板块查询，可能失败或较慢。
    所以这个函数设计成“尽力而为”：
        能匹配到就补 Sector；
        匹配不到就留空；
        不影响基础候选池。
    """
    if candidates is None or candidates.empty:
        return pd.DataFrame()

    frame = candidates.copy()
    frame["Sector"] = ""
    frame["SectorChangePct"] = None

    if sector_snapshot is None or sector_snapshot.empty:
        return frame

    for _, sector in sector_snapshot.iterrows():
        sector_name = str(sector.get("Name") or "").strip()
        if not sector_name:
            continue

        members = get_industry_board_members(sector_name)
        if members.empty or "Symbol" not in members.columns:
            continue

        member_symbols = set(members["Symbol"].dropna().astype(str))
        matched = frame["Symbol"].isin(member_symbols) & (frame["Sector"] == "")
        frame.loc[matched, "Sector"] = sector_name
        frame.loc[matched, "SectorChangePct"] = sector.get("ChangePct")

    return frame


def get_industry_board_members(sector_name: str) -> pd.DataFrame:
    """获取行业板块成分股。

    如果接口失败，返回空表。
    """
    import akshare as ak

    try:
        raw = ak.stock_board_industry_cons_em(symbol=sector_name)
    except Exception:
        return pd.DataFrame(columns=["Symbol", "Name"])

    code_column = find_first_existing_column(raw, ["代码", "code", "symbol"])
    name_column = find_first_existing_column(raw, ["名称", "name"])
    if code_column is None:
        return pd.DataFrame(columns=["Symbol", "Name"])

    frame = pd.DataFrame()
    frame["Symbol"] = raw[code_column].map(safe_normalize_symbol)
    frame["Name"] = raw[name_column].map(lambda value: str(value).strip()) if name_column else ""
    return frame.dropna(subset=["Symbol"]).reset_index(drop=True)


def render_stock_screening_text(candidates: pd.DataFrame) -> str:
    """把候选股票池渲染成文本。"""
    if candidates is None or candidates.empty:
        return "暂无符合条件的候选股票。"

    return "\n\n".join(
        [
            "# A 股候选股票池原材料",
            "说明：以下候选来自全市场行情快照筛选，不等于最终买入建议。",
            candidates.fillna("").to_markdown(index=False),
        ]
    )


def find_first_existing_column(data: pd.DataFrame, candidates: list[str]) -> str | None:
    """从候选列中找到第一个存在的字段。"""
    for column in candidates:
        if column in data.columns:
            return column
    return None


def safe_normalize_symbol(value: Any) -> str | None:
    """安全标准化股票代码。"""
    try:
        return normalize_quote_symbol(value)
    except Exception:
        return None
