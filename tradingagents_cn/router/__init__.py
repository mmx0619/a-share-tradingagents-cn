"""自然语言路由层。

这里负责把人的问题转换成程序可以执行的结构化请求。
"""

from tradingagents_cn.router.user_question_router import (
    StockRouteItem,
    UserQuestionIntent,
    UserQuestionRoute,
    route_user_question,
)
from tradingagents_cn.router.llm_question_router import (
    LLMQuestionRouter,
    LLMRouteDecision,
    route_user_question_with_llm,
)

__all__ = [
    "LLMQuestionRouter",
    "LLMRouteDecision",
    "StockRouteItem",
    "UserQuestionIntent",
    "UserQuestionRoute",
    "route_user_question_with_llm",
    "route_user_question",
]
