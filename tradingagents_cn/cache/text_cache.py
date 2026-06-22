"""本地文本缓存工具。

这个文件解决一类问题：

    有些数据不是实时变化的，
    例如公司概况、资产负债表、现金流量表、利润表。

这些数据第一次联网获取后，可以先保存到本地。
下一次分析同一只股票时，如果缓存还没有过期，
就直接读取本地文件，避免反复爬取同一份低频数据。

注意：
    这里不是交易决策记忆。
    它只是“数据缓存层”，负责缓存从外部网站拿到的原材料文本。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "outputs" / "cache"


@dataclass
class TextCacheResult:
    """文本缓存读取结果。

    text:
        最终返回给上层使用的文本。

    cache_hit:
        True 表示本次直接使用了本地缓存；
        False 表示本次调用了外部数据源，并重新写入缓存。

    path:
        缓存文件路径，方便调试。
    """

    text: str
    cache_hit: bool
    path: Path


def get_or_refresh_text_cache(
    cache_group: str,
    cache_key: str,
    fetcher: Callable[[], str],
    max_age_days: float,
    force_refresh: bool = False,
    cache_root: str | Path | None = None,
) -> TextCacheResult:
    """读取文本缓存；缓存不存在或过期时，重新获取并覆盖。

    参数：
        cache_group:
            缓存分组，例如 fundamentals / news。

        cache_key:
            缓存键，例如 000725_balance_sheet_rows4_cols30。

        fetcher:
            真正联网获取数据的函数。
            只有缓存不存在、过期、或者 force_refresh=True 时才会调用。

        max_age_days:
            缓存有效天数。
            例如财务表按季度更新，可以设置成 35 天左右。

        force_refresh:
            是否强制忽略缓存，直接联网刷新。

        cache_root:
            缓存根目录。
            不传时默认使用：
                项目目录/outputs/cache
    """
    path = build_cache_path(
        cache_group=cache_group,
        cache_key=cache_key,
        cache_root=cache_root,
    )

    if not force_refresh:
        cached = read_cache_if_fresh(path, max_age_days=max_age_days)
        if cached is not None:
            return TextCacheResult(
                text=cached,
                cache_hit=True,
                path=path,
            )

    text = fetcher()
    write_text_cache(path, text)
    return TextCacheResult(
        text=text,
        cache_hit=False,
        path=path,
    )


def build_cache_path(
    cache_group: str,
    cache_key: str,
    cache_root: str | Path | None = None,
) -> Path:
    """根据缓存分组和缓存键生成 JSON 文件路径。"""
    root = Path(
        cache_root
        or os.getenv("TRADINGAGENTS_CN_CACHE_DIR")
        or DEFAULT_CACHE_ROOT
    )
    safe_group = sanitize_path_part(cache_group)
    safe_key = hash_cache_key(cache_key)
    return root / safe_group / f"{safe_key}.json"


def read_cache_if_fresh(path: Path, max_age_days: float) -> str | None:
    """如果缓存存在且未过期，返回缓存文本；否则返回 None。"""
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    cached_at_text = payload.get("cached_at")
    text = payload.get("text")
    if not cached_at_text or not isinstance(text, str):
        return None

    try:
        cached_at = datetime.fromisoformat(cached_at_text)
    except ValueError:
        return None

    age_days = (now_utc() - cached_at).total_seconds() / 86400
    if age_days > max_age_days:
        return None

    return text


def write_text_cache(path: Path, text: str) -> None:
    """把文本写入缓存文件。

    写入格式使用 JSON，是为了后续可以继续加元数据，
    例如数据来源、最新报告期、接口名称等。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cached_at": now_utc().isoformat(),
        "text": text,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def hash_cache_key(cache_key: str) -> str:
    """把可读缓存键转换成短文件名。

    为什么不直接把 cache_key 当文件名？
        因为 cache_key 里可能包含中文、斜杠、空格、冒号等字符。
        用 hash 可以避免 Windows 文件名问题。
    """
    normalized = str(cache_key or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def sanitize_path_part(text: str) -> str:
    """清理路径片段，避免生成非法目录名。"""
    invalid_chars = '<>:"/\\|?*'
    cleaned = str(text or "").strip()
    for char in invalid_chars:
        cleaned = cleaned.replace(char, "_")
    return cleaned or "default"


def now_utc() -> datetime:
    """返回带时区的 UTC 当前时间，便于计算缓存年龄。"""
    return datetime.now(timezone.utc)
