"""大模型接口错误提示工具。

这里不处理行情、新闻、财务数据错误，只处理模型 HTTP 接口返回的错误。
目的很简单：
    不让用户看到一大段 requests / LangGraph traceback；
    而是看到“该去检查 API Key、余额、模型名还是网络”的明确提示。
"""

from __future__ import annotations

from typing import Any


class LLMAPIError(RuntimeError):
    """大模型 API 调用失败。

    这类错误通常不是继续重试就能解决的业务配置问题，
    例如余额不足、API Key 错误、模型名错误。
    """


def build_llm_http_error_message(provider: str, error: Any) -> str:
    """把 requests.HTTPError 转成面向用户的中文说明。"""
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    reason = getattr(response, "reason", "") or ""
    body = safe_response_body(response)
    provider_name = provider.strip() or "模型服务商"

    if status_code == 400:
        hint = "请求参数不合法。常见原因是模型名、上下文长度、工具调用格式或 max_tokens 参数不符合该服务商要求。"
    elif status_code == 401:
        hint = "API Key 无效、过期或没有正确配置环境变量。请检查对应服务商的 API Key。"
    elif status_code == 402:
        hint = "账户余额不足、未开通计费、额度不可用，或该账号暂时不能继续调用付费 API。请到服务商控制台检查充值、账单和额度。"
    elif status_code == 403:
        hint = "当前 API Key 没有权限访问这个模型或接口。请检查模型权限、账号权限和服务商控制台设置。"
    elif status_code == 404:
        hint = "接口地址或模型名不存在。请检查 base_url 和 model。"
    elif status_code == 429:
        hint = "触发限流或额度限制。程序已经重试过，仍然失败；可以稍后再试或降低调用频率。"
    elif status_code is not None and 500 <= int(status_code) <= 599:
        hint = "服务商服务器临时错误。程序已经重试过，仍然失败；可以稍后再试。"
    else:
        hint = "模型接口返回了非成功状态。请检查服务商控制台、API Key、模型名和网络。"

    lines = [
        f"{provider_name} API 调用失败：HTTP {status_code or '未知'} {reason}".strip(),
        f"原因判断：{hint}",
    ]
    if body:
        lines.append(f"接口返回：{body}")
    return "\n".join(lines)


def safe_response_body(response: Any, max_chars: int = 500) -> str:
    """安全提取接口返回正文，避免把超长 HTML/JSON 全部打印到终端。"""
    if response is None:
        return ""
    try:
        text = str(getattr(response, "text", "") or "").strip()
    except Exception:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text
