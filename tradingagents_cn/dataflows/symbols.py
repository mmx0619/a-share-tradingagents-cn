"""A 股代码和日期格式工具。

这个文件属于正式工程的 dataflows 数据层。

它不负责联网，也不负责调用大模型。

它只负责一些最基础、但后面到处都会用到的格式转换：

1. 把股票代码统一成 6 位数字。
2. 判断股票属于哪个市场，并加上 sh / sz / bj 前缀。
3. 把日期从 YYYY-MM-DD 转成 AKShare 常用的 YYYYMMDD。

为什么要单独放一个文件？

因为行情、新闻、公告、财报、技术指标都要处理股票代码。
如果每个文件都自己写一遍代码清洗逻辑，后面很容易不一致。

例如：
    一个文件支持 600519.SH；
    另一个文件只支持 600519；
    第三个文件又支持 SH600519。

这样项目越大越难调试。

所以正式工程里先把“代码标准化”抽出来。
"""

from __future__ import annotations

from datetime import datetime


def normalize_cn_symbol(symbol: str) -> str:
    """把 A 股代码统一成 6 位数字格式。

    这个函数接收用户可能输入的各种写法，然后统一返回 6 位数字。

    支持示例：
        600519      -> 600519
        600519.SH   -> 600519
        000001.SZ   -> 000001
        SH600519    -> 600519
        sz002361    -> 002361
        BJ430047    -> 430047

    为什么要统一成 6 位数字？

    因为 AKShare 很多 A 股接口要求传入：
        600519
        002361

    而不是：
        SH600519
        600519.SH

    如果格式不统一，后面每个数据接口都要单独判断，会很乱。
    """
    cleaned = str(symbol or "").strip().upper()

    # 处理后缀格式：
    #   600519.SH
    #   000001.SZ
    #   430047.BJ
    for suffix in (".SH", ".SZ", ".BJ"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break

    # 处理前缀格式：
    #   SH600519
    #   SZ002361
    #   BJ430047
    #
    # 正常情况下前缀 2 位 + 股票代码 6 位 = 总长度 8。
    for prefix in ("SH", "SZ", "BJ"):
        if cleaned.startswith(prefix) and len(cleaned) == 8:
            cleaned = cleaned[2:]
            break

    if len(cleaned) != 6 or not cleaned.isdigit():
        raise ValueError(f"不是合法的 A 股 6 位股票代码：{symbol!r}")

    return cleaned


def to_market_prefixed_symbol(symbol: str) -> str:
    """把 6 位股票代码转换成带市场前缀的格式。

    有些接口需要这种格式：
        sh600519
        sz002361
        bj430047

    判断规则：
        6、9 开头：通常是上交所，使用 sh。
        0、2、3 开头：通常是深交所，使用 sz。
        4、8 开头：通常是北交所，使用 bj。

    注意：
        这里是常见公开行情接口的工程规则，
        不是交易所完整证券分类规则。
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
    """把日期转换成 AKShare 常用的 YYYYMMDD 格式。

    用户和工程里更容易阅读的日期格式是：
        2026-06-17

    AKShare 的很多接口需要：
        20260617

    所以这里统一转换。

    如果传入的日期格式不对，datetime.strptime 会抛出 ValueError，
    这样调用方能尽早发现问题。
    """
    return datetime.strptime(date_text, "%Y-%m-%d").strftime("%Y%m%d")
