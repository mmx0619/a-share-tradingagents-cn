"""大模型用户问题路由。

这个文件负责让大模型理解用户自然语言问题，
并提取程序后续执行需要的信息。

为什么需要大模型路由？

规则路由能识别：
    大唐发电行情怎么样
    601991 怎么样

但用户可能会问得更口语：
    火电那个大唐现在咋样
    面板龙头还能不能上车
    宁王是不是跌出机会了

这种问题靠字符串匹配很难覆盖。

所以当前推荐流程是：

    大模型结构化路由
      -> 程序校验
      -> 规则路由兜底
      -> 调用具体分析链路
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from tradingagents_cn.dataflows.stock_directory import (
    StockSymbolMatch,
    find_stock_name_by_symbol,
    resolve_stock_from_text,
    resolve_stocks_from_text,
)
from tradingagents_cn.dataflows.symbols import normalize_cn_symbol
from tradingagents_cn.llm.factory import create_chat_client
from tradingagents_cn.llm.structured_output import call_structured_output
from tradingagents_cn.router.user_question_router import (
    UserQuestionIntent,
    UserQuestionRoute,
    build_multi_stock_route,
    build_single_stock_route,
    route_user_question,
)


class RouterConfidence(str, Enum):
    """路由置信度。"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LLMStockItem(BaseModel):
    """大模型识别出的单只股票。"""

    stock_name: str | None = Field(default=None, description="股票名称。")
    symbol: str | None = Field(default=None, description="6 位 A 股代码。")


class LLMRouteDecision(BaseModel):
    """大模型路由结构化输出。

    intent:
        用户问题意图，必须是固定枚举。

    stock_name:
        用户提到的股票名称。
        如果没有明确股票，返回 null。

    symbol:
        6 位 A 股代码。
        如果没有明确股票，返回 null。

    stock_items:
        用户一次提到多只股票时，按提问顺序返回股票列表。

    question_focus:
        用户关注点，例如“能不能买”“行情怎么样”“今天大盘”。

    confidence:
        模型对识别结果的置信度。

    reason:
        简要说明为什么这样识别。
    """

    intent: Literal[
        "single_stock_analysis",
        "multi_stock_analysis",
        "market_overview",
        "stock_screening",
        "out_of_scope",
        "unknown",
    ] = Field(description="用户问题意图。")
    stock_name: str | None = Field(default=None, description="股票名称。")
    symbol: str | None = Field(default=None, description="6 位 A 股代码。")
    stock_items: list[LLMStockItem] = Field(default_factory=list, description="多股问题中的股票列表。")
    question_focus: str = Field(default="", description="用户关注点。")
    confidence: Literal["high", "medium", "low"] = Field(description="置信度。")
    reason: str = Field(description="识别理由。")


@dataclass
class LLMQuestionRouter:
    """大模型问题路由器。"""

    llm_client: Any | None = None
    temperature: float = 0.0

    def route(self, question: str) -> UserQuestionRoute:
        """用大模型识别用户问题，并返回 UserQuestionRoute。

        如果大模型输出无法解析或校验失败，
        自动回退到规则路由 route_user_question(...)。
        """
        original_question = str(question or "").strip()
        if not original_question:
            return UserQuestionRoute(
                intent=UserQuestionIntent.UNKNOWN,
                original_question=original_question,
                reason="用户问题为空。",
            )

        fast_multi_matches = resolve_stocks_from_text(original_question, use_akshare=False)
        if len(fast_multi_matches) >= 2:
            route = build_multi_stock_route(original_question, fast_multi_matches)
            route.reason = "本地别名表快速识别到多只股票，直接进入多股完整分析。"
            return route

        try:
            decision = self.invoke_llm_router(original_question)
            route = convert_llm_decision_to_route(original_question, decision)
            if route.intent == UserQuestionIntent.UNKNOWN:
                repaired_route = repair_unknown_route_with_stock_directory(original_question, route)
                if repaired_route.intent != UserQuestionIntent.UNKNOWN:
                    return repaired_route
            return route
        except Exception as error:
            fallback_route = route_user_question(original_question)
            fallback_route.reason = (
                f"LLM 路由失败，已回退到规则路由。错误：{error}；"
                f"规则路由原因：{fallback_route.reason}"
            )
            return fallback_route

    def invoke_llm_router(self, question: str) -> LLMRouteDecision:
        """调用大模型并解析结构化路由结果。"""
        client = self.llm_client or create_chat_client()
        result = call_structured_output(
            llm_client=client,
            messages=build_llm_router_messages(question),
            schema_model=LLMRouteDecision,
            fallback_factory=build_fallback_llm_route_decision,
            temperature=self.temperature,
            max_retries=2,
        )
        return result.value


def build_llm_router_messages(question: str) -> list[dict[str, str]]:
    """构造大模型路由 messages。"""
    schema = LLMRouteDecision.model_json_schema()
    return [
        {
            "role": "system",
            "content": (
                "你是 A 股投研助手的用户问题路由器。"
                "你只负责理解用户问题，不做股票分析。"
                "你必须输出合法 JSON，不要输出 Markdown。"
                "如果用户提到单只 A 股股票，无论他说行情、能不能买、走势、怎么看，"
                "都路由为 single_stock_analysis。"
                "如果用户一次提到多只 A 股股票，路由为 multi_stock_analysis，"
                "并把所有股票按用户提问顺序放入 stock_items。"
                "如果用户问大盘、今天股市、市场整体，路由为 market_overview。"
                "如果用户让你推荐股票、筛股、找机会，路由为 stock_screening。"
                "如果用户问题与股票、A 股、市场、投资研究无关，路由为 out_of_scope。"
                "如果无法判断，路由为 unknown。"
                "如果你知道股票代码，填写 6 位 symbol；如果不知道，symbol 返回 null。"
                "多股问题中，如果你知道每只股票代码，就在 stock_items 里填写；"
                "如果不知道某只代码，该项 symbol 返回 null。"
                "JSON Schema 如下："
                f"{json.dumps(schema, ensure_ascii=False)}"
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]


def parse_llm_route_decision(text: str) -> LLMRouteDecision:
    """解析并校验大模型路由输出。"""
    json_text = extract_json_object_text(text)
    data = json.loads(json_text)
    return LLMRouteDecision.model_validate(data)


def convert_llm_decision_to_route(
    original_question: str,
    decision: LLMRouteDecision,
) -> UserQuestionRoute:
    """把 LLMRouteDecision 转成工程内通用 UserQuestionRoute。

    这里会做校验和补全：
        - 如果 symbol 存在，必须是合法 6 位代码；
        - 如果 symbol 不存在但 stock_name 存在，尝试用规则目录补全；
        - 如果单股问题仍没有 symbol，降级为 unknown，避免乱跑。
    """
    intent = UserQuestionIntent(decision.intent)

    if intent == UserQuestionIntent.MULTI_STOCK_ANALYSIS:
        stock_matches = normalize_llm_stock_items(decision.stock_items)
        if len(stock_matches) < 2:
            stock_matches = resolve_stocks_for_router(original_question)

        if len(stock_matches) >= 2:
            route = build_multi_stock_route(original_question, stock_matches)
            route.reason = (
                "LLM 结构化路由识别为多股完整分析。"
                f"confidence={decision.confidence}；"
                f"focus={decision.question_focus}；"
                f"reason={decision.reason}；"
                f"校验后股票数量={len(stock_matches)}。"
            )
            return route

        if len(stock_matches) == 1:
            return build_single_stock_route(original_question, stock_matches[0])

        return UserQuestionRoute(
            intent=UserQuestionIntent.UNKNOWN,
            original_question=original_question,
            analysis_depth="full",
            reason="LLM 判断为多股分析，但没有识别出至少两只可校验股票。",
        )

    if intent == UserQuestionIntent.SINGLE_STOCK_ANALYSIS:
        symbol = normalize_symbol_or_none(decision.symbol)
        stock_name = normalize_text_or_none(decision.stock_name)

        if symbol is None and stock_name:
            matched = resolve_stock_from_text(stock_name, use_akshare=False)
            if matched is None:
                matched = resolve_stock_from_text(stock_name, use_akshare=True)
            if matched is not None:
                symbol = matched.symbol
                stock_name = matched.stock_name

        if symbol is None:
            stock_matches = resolve_stocks_for_router(original_question)
            if len(stock_matches) >= 2:
                route = build_multi_stock_route(original_question, stock_matches)
                route.reason = (
                    "LLM 判断为单股分析但未给出代码；"
                    "程序从原问题中识别到多只股票，已修正为多股完整分析。"
                )
                return route
            if len(stock_matches) == 1:
                return build_single_stock_route(original_question, stock_matches[0])

            return UserQuestionRoute(
                intent=UserQuestionIntent.UNKNOWN,
                original_question=original_question,
                stock_name=stock_name,
                analysis_depth="full",
                reason=(
                    "LLM 判断为单股分析，但没有给出可校验的 6 位股票代码，"
                    "因此不执行完整链路。"
                ),
            )

        if stock_name is None:
            stock_name = find_stock_name_by_symbol(symbol) or symbol

        return UserQuestionRoute(
            intent=UserQuestionIntent.SINGLE_STOCK_ANALYSIS,
            original_question=original_question,
            symbol=symbol,
            stock_name=stock_name,
            analysis_depth="full",
            reason=(
                "LLM 结构化路由识别为单股完整分析。"
                f"confidence={decision.confidence}；"
                f"focus={decision.question_focus}；"
                f"reason={decision.reason}"
            ),
        )

    return UserQuestionRoute(
        intent=intent,
        original_question=original_question,
        stock_name=normalize_text_or_none(decision.stock_name),
        symbol=normalize_symbol_or_none(decision.symbol),
        analysis_depth="full",
        reason=(
            "LLM 结构化路由。"
            f"confidence={decision.confidence}；"
            f"focus={decision.question_focus}；"
            f"reason={decision.reason}"
        ),
    )


def normalize_llm_stock_items(stock_items: list[LLMStockItem]) -> list[StockSymbolMatch]:
    """校验大模型返回的多股列表。

    只把能落到 6 位 A 股代码的项目交给后续链路。
    """
    matches: list[StockSymbolMatch] = []
    seen_symbols: set[str] = set()

    for item in stock_items:
        symbol = normalize_symbol_or_none(item.symbol)
        stock_name = normalize_text_or_none(item.stock_name)

        if symbol is None and stock_name:
            matched = resolve_stock_from_text(stock_name, use_akshare=False)
            if matched is None:
                matched = resolve_stock_from_text(stock_name, use_akshare=True)
            if matched is not None:
                symbol = matched.symbol
                stock_name = matched.stock_name

        if symbol is None:
            continue

        if stock_name is None:
            stock_name = find_stock_name_by_symbol(symbol) or symbol

        if symbol in seen_symbols:
            continue
        seen_symbols.add(symbol)
        matches.append(
            StockSymbolMatch(
                symbol=symbol,
                stock_name=stock_name,
                match_source="llm_router",
            )
        )

    return matches


def repair_unknown_route_with_stock_directory(
    original_question: str,
    route: UserQuestionRoute,
) -> UserQuestionRoute:
    """当 LLM 路由失败为 unknown 时，再用股票目录兜底修复。

    这一步专门处理：
        用户问了多只股票，但模型没有按 schema 填好 stock_items。
    """
    stock_matches = resolve_stocks_for_router(original_question)
    if len(stock_matches) >= 2:
        repaired = build_multi_stock_route(original_question, stock_matches)
        repaired.reason = f"{route.reason}；程序用股票目录修复为多股完整分析。"
        return repaired
    if len(stock_matches) == 1:
        repaired = build_single_stock_route(original_question, stock_matches[0])
        repaired.reason = f"{route.reason}；程序用股票目录修复为单股完整分析。"
        return repaired
    return route


def resolve_stocks_for_router(original_question: str) -> list[StockSymbolMatch]:
    """路由层识别多只股票。

    先用本地别名表，速度快，也不依赖网络；
    如果本地只识别出 0 或 1 只，再调用 AKShare 股票名称表补全。
    """
    local_matches = resolve_stocks_from_text(original_question, use_akshare=False)
    if len(local_matches) >= 2:
        return local_matches

    akshare_matches = resolve_stocks_from_text(original_question, use_akshare=True)
    if len(akshare_matches) > len(local_matches):
        return akshare_matches
    return local_matches


def normalize_symbol_or_none(symbol: str | None) -> str | None:
    """校验并标准化股票代码。"""
    if not symbol:
        return None
    try:
        return normalize_cn_symbol(symbol)
    except Exception:
        return None


def normalize_text_or_none(text: str | None) -> str | None:
    """清洗可选文本。"""
    cleaned = str(text or "").strip()
    return cleaned or None


def extract_json_object_text(text: str) -> str:
    """从模型输出里提取 JSON 对象。"""
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.S)
    if fenced_match:
        return fenced_match.group(1)

    object_match = re.search(r"\{.*\}", stripped, re.S)
    if object_match:
        return object_match.group(0)

    raise ValueError("模型输出中没有找到 JSON 对象。")


def route_user_question_with_llm(
    question: str,
    llm_client: Any | None = None,
    temperature: float = 0.0,
) -> UserQuestionRoute:
    """函数式入口：使用大模型路由用户问题。"""
    router = LLMQuestionRouter(llm_client=llm_client, temperature=temperature)
    return router.route(question)


def build_fallback_llm_route_decision(error_message: str) -> LLMRouteDecision:
    """构造 LLM Router 的合法兜底对象。"""
    return LLMRouteDecision(
        intent="unknown",
        stock_name=None,
        symbol=None,
        stock_items=[],
        question_focus="",
        confidence="low",
        reason=f"模型路由结构化输出失败，进入规则路由兜底。错误：{error_message}",
    )
