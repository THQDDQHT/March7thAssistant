# coding:utf-8
"""第二期：DeepSeek Search / Fetch 客户端（mock HTTP）：解析、重试、错误转换、key 不泄露、分段。"""
import pytest
import requests

from module.llm_agent import search as sr

KEY = "sk-test-SECRET-1234567890"


class FakeResponse:
    def __init__(self, status=200, data=None, text=""):
        self.status_code = status
        self._data = data
        self.text = text or (str(data) if data is not None else "")

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data


class FakeHttp:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make(http, **kwargs):
    sleeps = []
    client = sr.DeepSeekSearch(api_key=KEY, http=http, sleep=sleeps.append, **kwargs)
    return client, sleeps


SEARCH_OK = {"results": [
    {"title": "5.0 版本活动", "url": "https://a.example/1", "snippet": "片段", "summary": "总结", "content": "全文",
     "site_name": "示例站", "date": "2026-10-30", "rerank_score": 0.9, "authority": 3},
    {"title": "只有总结", "url": "https://a.example/2", "summary": "总结内容"},
    "not a dict",
]}


class TestSearch:
    def test_request_shape_and_parsing(self):
        http = FakeHttp(FakeResponse(200, SEARCH_OK))
        client, _ = make(http)
        results = client.search("  崩坏星穹铁道 当前版本活动 ", 5)
        call = http.calls[0]
        assert call["url"] == "https://api.deepseek.com/v1/tools/search"
        assert call["headers"]["Authorization"] == f"Bearer {KEY}"
        assert call["json"] == {"query": "崩坏星穹铁道 当前版本活动", "count": 5, "need_content": False}
        assert [r.title for r in results] == ["5.0 版本活动", "只有总结"]
        assert results[0].snippet == "片段" and results[0].site_name == "示例站" and results[0].score == 0.9
        assert results[1].snippet == "总结内容"  # 没有 snippet 时退回 summary

    def test_nested_data_results(self):
        client, _ = make(FakeHttp(FakeResponse(200, {"data": {"results": [{"title": "t", "url": "u"}]}})))
        assert client.search("x")[0].title == "t"

    def test_missing_results_is_error(self):
        client, _ = make(FakeHttp(FakeResponse(200, {"foo": 1})))
        with pytest.raises(sr.SearchError, match="results"):
            client.search("x")

    @pytest.mark.parametrize("query", ["", "   ", None, 5])
    def test_empty_query(self, query):
        client, _ = make(FakeHttp())
        with pytest.raises(sr.SearchError, match="query 不能为空"):
            client.search(query)

    def test_query_length_limit(self):
        client, _ = make(FakeHttp())
        with pytest.raises(sr.SearchError, match="128"):
            client.search("字" * 129)
        client2, _ = make(FakeHttp(FakeResponse(200, {"results": []})))
        assert client2.search("字" * 128) == []

    @pytest.mark.parametrize("count", [0, 51, -1, "abc"])
    def test_count_range(self, count):
        client, _ = make(FakeHttp())
        with pytest.raises(sr.SearchError, match="count"):
            client.search("x", count)

    def test_no_api_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        client = sr.DeepSeekSearch(http=FakeHttp())
        with pytest.raises(sr.SearchError, match="DEEPSEEK_API_KEY"):
            client.search("x")

    def test_key_is_read_from_environment_at_call_time(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
        http = FakeHttp(FakeResponse(200, {"results": []}))
        sr.DeepSeekSearch(http=http).search("x")
        assert http.calls[0]["headers"]["Authorization"] == f"Bearer {KEY}"


class TestRetryAndErrors:
    @pytest.mark.parametrize("status", [408, 429, 500, 502, 503])
    def test_retry_then_success(self, status):
        http = FakeHttp(FakeResponse(status), FakeResponse(status), FakeResponse(200, {"results": []}))
        client, sleeps = make(http)
        assert client.search("x") == []
        assert len(http.calls) == 3 and sleeps == [1.0, 2.0]

    def test_retry_exhausted(self):
        http = FakeHttp(FakeResponse(429), FakeResponse(429), FakeResponse(429))
        client, _ = make(http)
        with pytest.raises(sr.SearchError, match="重试 2 次后仍然失败.*429"):
            client.search("x")
        assert len(http.calls) == 3

    @pytest.mark.parametrize("status,message", [
        (400, "请求参数不合法"), (401, "API key 无效"), (402, "余额不足"), (403, "没有权限"),
        (404, "不存在"), (413, "过大")])
    def test_no_retry_for_client_errors(self, status, message):
        http = FakeHttp(FakeResponse(status, text="bad"))
        client, sleeps = make(http)
        with pytest.raises(sr.SearchError, match=message):
            client.search("x")
        assert len(http.calls) == 1 and sleeps == []

    def test_network_error_retries(self):
        http = FakeHttp(requests.ConnectionError("boom"), FakeResponse(200, {"results": []}))
        client, sleeps = make(http)
        assert client.search("x") == []
        assert sleeps == [1.0]

    def test_network_error_exhausted_message_has_no_details(self):
        err = requests.ConnectionError(f"https://x?key={KEY}")
        client, _ = make(FakeHttp(err, err, err))
        with pytest.raises(sr.SearchError) as info:
            client.search("x")
        assert KEY not in str(info.value) and "ConnectionError" in str(info.value)

    def test_key_in_error_body_is_redacted(self):
        http = FakeHttp(FakeResponse(400, text=f"invalid request for {KEY}"))
        client, _ = make(http)
        with pytest.raises(sr.SearchError) as info:
            client.search("x")
        assert KEY not in str(info.value) and "***" in str(info.value)

    def test_non_json_and_non_dict_response(self):
        client, _ = make(FakeHttp(FakeResponse(200, None, text="<html>")))
        with pytest.raises(sr.SearchError, match="JSON"):
            client.search("x")
        client, _ = make(FakeHttp(FakeResponse(200, ["a"])))
        with pytest.raises(sr.SearchError, match="格式"):
            client.search("x")

    def test_repr_has_no_key(self):
        assert KEY not in repr(make(FakeHttp())[0])


class TestFetch:
    def test_parse(self):
        http = FakeHttp(FakeResponse(200, {"title": "标题", "content": "# 正文"}))
        client, _ = make(http)
        page = client.fetch(" https://a.example/1 ")
        assert http.calls[0]["url"] == "https://api.deepseek.com/v1/tools/fetch"
        assert http.calls[0]["json"] == {"url": "https://a.example/1"}
        assert (page.title, page.content, page.url) == ("标题", "# 正文", "https://a.example/1")

    @pytest.mark.parametrize("url", ["", "ftp://x", "javascript:alert(1)", "a.example/1", None])
    def test_url_validation(self, url):
        client, _ = make(FakeHttp())
        with pytest.raises(sr.SearchError):
            client.fetch(url)

    def test_missing_content(self):
        client, _ = make(FakeHttp(FakeResponse(200, {"title": "t"})))
        with pytest.raises(sr.SearchError, match="content"):
            client.fetch("https://a.example/1")


class TestFormatting:
    def test_format_results(self):
        results = [sr.SearchResult("标题", "https://a", "很长的摘要" * 100, "站点", "2026-10-30")]
        text = sr.format_results(results, "查询")
        assert "标题" in text and "链接：https://a" in text and "来源：站点 · 2026-10-30" in text
        assert text.count("很长的摘要") < 100 and "…" in text

    def test_format_empty(self):
        assert "没有结果" in sr.format_results([], "查询")

    def test_slice_content(self):
        assert sr.slice_content("0123456789", 0, 4) == ("0123", 10, 4)
        assert sr.slice_content("0123456789", 8, 4) == ("89", 10, None)
        assert sr.slice_content("0123456789", 10, 4) == ("", 10, None)
