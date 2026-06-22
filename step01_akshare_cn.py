"""第 01 步：用 AKShare 获取 A 股日线行情。

这个文件故意保持独立，暂时不依赖 TradingAgents 项目。

它目前只做三件事：
1. 把常见 A 股代码格式统一成 6 位数字代码。
2. 通过 AKShare 获取 A 股历史日线数据。
3. 把 AKShare 返回的中文列名转换成简单的英文 OHLCV 列名。
"""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd


COLUMN_MAP = {
    "日期": "Date",
    "开盘": "Open",
    "最高": "High",
    "最低": "Low",
    "收盘": "Close",
    "成交量": "Volume",
    "成交额": "Amount",
    "换手率": "Turnover",
    # 腾讯证券备用接口返回的是小写英文字段。
    "date": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "amount": "Volume",
}


def normalize_cn_symbol(symbol: str) -> str:
    """把 A 股代码统一成 AKShare 需要的 6 位数字格式。

    示例：
    - 600519 -> 600519
    - 600519.SH -> 600519
    - SH600519 -> 600519
    - sz000001 -> 000001
    """
    cleaned = str(symbol or "").strip().upper()

    for suffix in (".SH", ".SZ", ".BJ"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break

    for prefix in ("SH", "SZ", "BJ"):
        if cleaned.startswith(prefix) and len(cleaned) == 8:
            cleaned = cleaned[2:]
            break

    if len(cleaned) != 6 or not cleaned.isdigit():
        raise ValueError(f"Invalid A-share symbol: {symbol!r}")

    return cleaned


def to_market_prefixed_symbol(symbol: str) -> str:
    """把 6 位股票代码转换成腾讯接口需要的带市场前缀格式。

    示例：
    - 600519 -> sh600519
    - 000001 -> sz000001
    - 002361 -> sz002361
    - 300750 -> sz300750
    - 688001 -> sh688001
    - 北交所 8 开头股票 -> bjxxxxxx
    """
    normalized_symbol = normalize_cn_symbol(symbol)

    if normalized_symbol.startswith(("6", "9")):
        return f"sh{normalized_symbol}"
    if normalized_symbol.startswith(("0", "2", "3")):
        return f"sz{normalized_symbol}"
    if normalized_symbol.startswith(("4", "8")):
        return f"bj{normalized_symbol}"

    raise ValueError(f"无法判断股票代码所属市场：{symbol!r}")


def to_akshare_date(date_text: str) -> str:
    """把 YYYY-MM-DD 格式日期转换成 AKShare 需要的 YYYYMMDD 格式。"""
    return datetime.strptime(date_text, "%Y-%m-%d").strftime("%Y%m%d")


def normalize_history_frame(data: pd.DataFrame) -> pd.DataFrame:
    """把 AKShare 历史行情表转换成 Date/Open/High/Low/Close/Volume 格式。"""
    if data is None or data.empty:
        raise ValueError("AKShare returned empty history data.")

    frame = data.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in data.columns})

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    optional = [column for column in ("Amount", "Turnover") if column in frame.columns]
    frame = frame[required + optional].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"])

    for column in [column for column in frame.columns if column != "Date"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["Close"])
    return frame.sort_values("Date")


def get_a_share_daily_history(
    symbol: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    retry_sleep_seconds: float = 1.5,
) -> pd.DataFrame:
    """通过 AKShare 获取 A 股历史日线行情。

    调用这个函数前，需要先安装 AKShare：
    pip install akshare

    为什么要有重试：
    AKShare 底层访问的是东方财富等公开数据接口。
    这些接口偶尔会断开连接或临时失败。
    如果不重试，后面的 Agent 流水线会被一次网络抖动打断。
    """
    import akshare as ak

    normalized_symbol = normalize_cn_symbol(symbol)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            # 优先使用东方财富接口。
            # 它字段更完整，包含成交额、换手率等信息。
            raw = ak.stock_zh_a_hist(
                symbol=normalized_symbol,
                period="daily",
                start_date=to_akshare_date(start_date),
                end_date=to_akshare_date(end_date),
                adjust="qfq",
            )
            return normalize_history_frame(raw)
        except Exception as error:
            last_error = error
            if attempt < max_retries:
                time.sleep(retry_sleep_seconds)

    # 东方财富接口失败时，使用腾讯证券接口兜底。
    # 腾讯接口字段少一点，但足够计算 OHLCV 技术指标。
    try:
        raw = ak.stock_zh_a_hist_tx(
            symbol=to_market_prefixed_symbol(normalized_symbol),
            start_date=to_akshare_date(start_date),
            end_date=to_akshare_date(end_date),
            adjust="qfq",
            timeout=15,
        )
        return normalize_history_frame(raw)
    except Exception as fallback_error:
        raise RuntimeError(
            f"AKShare 获取 {normalized_symbol} 日线行情失败。"
            f"东方财富已重试 {max_retries} 次，最后错误：{last_error}；"
            f"腾讯备用接口错误：{fallback_error}"
        ) from fallback_error


if __name__ == "__main__":
    df = get_a_share_daily_history("600519", "2024-01-01", "2024-01-10")
    print(df.head())
