"""DeepSeek Search / Fetch 客户端（第二期 MCP 工具 web_search / fetch_page 用）。

- Search：`POST {base}/v1/tools/search`，字段 query（1-128 字符）、count（1-50）、need_content 等；返回 results[]。
- Fetch：`POST {base}/v1/tools/fetch`，字段 url；返回 title、content（Markdown）。
- 错误码：400/401/402/403/404/408/413/429/5xx；其中 408、429、5xx 重试，其余直接报错。
- API key 只从环境变量 DEEPSEEK_API_KEY 读取，只出现在 Authorization 请求头里，
  不进入日志、异常信息和任何记录（所有出口文字都先过 redact）。
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import requests

from module.llm_agent.config import ENV_API_KEY, get_api_key, redact

DEFAULT_BASE_URL = "https://api.deepseek.com"
SEARCH_PATH = "/v1/tools/search"
FETCH_PATH = "/v1/tools/fetch"

QUERY_MAX_CHARS = 128
COUNT_MAX = 50
REQUEST_TIMEOUT = 30
RETRIES = 2
RETRY_DELAY = 1.0
RETRY_STATUS = (408, 429, 500, 502, 503, 504)

# 摘要在结果列表里的最大长度（需要全文请用 fetch_page）
SNIPPET_MAX = 240
# 错误信息里最多带回的响应正文长度
ERROR_BODY_MAX = 200

_STATUS_MESSAGES = {
    400: "请求参数不合法（400）",
    401: "API key 无效或未授权（401），请检查环境变量 DEEPSEEK_API_KEY",
    402: "账户余额不足（402）",
    403: "没有权限访问该接口（403）",
    404: "接口或网页不存在，或站点不允许抓取（404），换一个链接试试",
    408: "请求超时（408）",
    413: "请求或网页内容过大（413）",
    429: "请求过于频繁（429）",
}


class SearchError(Exception):
    """搜索或抓取失败；消息是给模型看的中文说明，不含 API key。"""


@dataclass
class SearchResult:
    title: str = ""
    url: str = ""
    snippet: str = ""
    site_name: str = ""
    date: str = ""
    score: Optional[float] = None


@dataclass
class FetchedPage:
    url: str
    title: str
    content: str


def validate_query(query) -> str:
    """校验搜索词（1-128 字符），返回去掉首尾空白的搜索词。"""
    if not isinstance(query, str) or not query.strip():
        raise SearchError("query 不能为空")
    query = query.strip()
    if len(query) > QUERY_MAX_CHARS:
        raise SearchError(f"query 太长（{len(query)} 字符），最多 {QUERY_MAX_CHARS} 字符，请精简关键词")
    return query


def validate_url(url) -> str:
    if not isinstance(url, str) or not url.strip():
        raise SearchError("url 不能为空")
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        raise SearchError("url 必须以 http:// 或 https:// 开头")
    if len(url) > 2048:
        raise SearchError("url 太长（超过 2048 字符）")
    return url


class DeepSeekSearch:
    def __init__(self, api_key: Optional[str] = None, base_url: str = DEFAULT_BASE_URL,
                 timeout: float = REQUEST_TIMEOUT, retries: int = RETRIES, http=None, sleep=time.sleep):
        self._api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self._http = http if http is not None else requests
        self._sleep = sleep

    def __repr__(self):
        return f"DeepSeekSearch(base_url={self.base_url!r})"

    def _key(self) -> str:
        # 每次调用时再读环境变量：服务启动后才设置的 key 也能生效
        key = self._api_key or get_api_key()
        if not key:
            raise SearchError(f"没有设置环境变量 {ENV_API_KEY}，无法联网搜索")
        return key

    def _safe(self, text, key: str) -> str:
        return redact(str(text), key)

    def _post(self, path: str, payload: dict) -> dict:
        key = self._key()
        url = self.base_url + path
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        last_error = "未知错误"
        for attempt in range(self.retries + 1):
            if attempt:
                self._sleep(RETRY_DELAY * attempt)
            try:
                response = self._http.post(url, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                last_error = f"网络错误：{type(e).__name__}"
                continue
            status = response.status_code
            if status in RETRY_STATUS:
                last_error = _STATUS_MESSAGES.get(status) or f"服务暂时不可用（HTTP {status}）"
                continue
            if status != 200:
                body = self._safe((getattr(response, "text", "") or "")[:ERROR_BODY_MAX], key)
                message = _STATUS_MESSAGES.get(status) or f"HTTP {status}"
                raise SearchError(f"{message}：{body}" if body else message)
            try:
                data = response.json()
            except ValueError:
                raise SearchError("响应不是合法的 JSON") from None
            if not isinstance(data, dict):
                raise SearchError("响应格式不对")
            return data
        raise SearchError(f"重试 {self.retries} 次后仍然失败：{last_error}")

    def search(self, query: str, count: int = 8, need_content: bool = False) -> List[SearchResult]:
        query = validate_query(query)
        try:
            count = int(count)
        except (TypeError, ValueError):
            raise SearchError("count 必须是 1-50 的整数") from None
        if not 1 <= count <= COUNT_MAX:
            raise SearchError(f"count 必须在 1-{COUNT_MAX} 之间")
        payload = {"query": query, "count": count, "need_content": bool(need_content)}
        data = self._post(SEARCH_PATH, payload)
        raw = data.get("results")
        if raw is None and isinstance(data.get("data"), dict):
            raw = data["data"].get("results")
        if not isinstance(raw, list):
            raise SearchError("响应里没有 results 列表")
        results = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            snippet = item.get("snippet") or item.get("summary") or ""
            score = item.get("rerank_score")
            results.append(SearchResult(
                title=str(item.get("title") or "").strip(),
                url=str(item.get("url") or "").strip(),
                snippet=str(snippet).strip(),
                site_name=str(item.get("site_name") or "").strip(),
                date=str(item.get("date") or "").strip(),
                score=float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
            ))
        return results

    def fetch(self, url: str) -> FetchedPage:
        url = validate_url(url)
        data = self._post(FETCH_PATH, {"url": url})
        content = data.get("content")
        if content is None and isinstance(data.get("data"), dict):
            data = data["data"]
            content = data.get("content")
        if not isinstance(content, str):
            raise SearchError("响应里没有网页正文（content）")
        return FetchedPage(url=url, title=str(data.get("title") or "").strip(), content=content)


# ------------------------------------------------------------- 格式化 ----

def format_results(results: List[SearchResult], query: str) -> str:
    """搜索结果 → 给模型看的文字：标题、链接、摘要、日期。"""
    if not results:
        return f"搜索「{query}」没有结果，换个关键词再试。"
    lines = [f"搜索「{query}」，共 {len(results)} 条结果（需要全文用 fetch_page 抓取链接）："]
    for index, item in enumerate(results, 1):
        lines.append(f"{index}. {item.title or '(无标题)'}")
        lines.append(f"   链接：{item.url}")
        meta = " · ".join(part for part in (item.site_name, item.date) if part)
        if meta:
            lines.append(f"   来源：{meta}")
        if item.snippet:
            snippet = " ".join(item.snippet.split())
            if len(snippet) > SNIPPET_MAX:
                snippet = snippet[:SNIPPET_MAX] + "…"
            lines.append(f"   摘要：{snippet}")
    return "\n".join(lines)


def slice_content(content: str, offset: int, max_chars: int) -> Tuple[str, int, Optional[int]]:
    """按 offset / max_chars 取一段正文，返回 (这一段, 总长度, 下一段的 offset；已经到末尾时为 None)。"""
    total = len(content)
    segment = content[offset:offset + max_chars]
    end = offset + len(segment)
    return segment, total, (end if end < total else None)
