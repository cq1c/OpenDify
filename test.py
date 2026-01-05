#!/usr/bin/env python3
"""
OpenDify - 全量功能测试（Unit + Integration）

运行：
- 仅单元测试：python test.py（无需启动服务）
- 集成测试：先启动服务 python app.py，再运行 python test.py

可选环境变量：
- TEST_SERVER_ORIGIN: 覆盖服务地址（默认从 SERVER_HOST/SERVER_PORT 推导）
- TEST_PROXY_API_KEY: 覆盖代理鉴权 key（默认取 VALID_API_KEYS 的第一个）
- TEST_DIFY_MODEL: 覆盖 Dify 模型名（默认从 /v1/models 自动选择第一个）
- TEST_CLAUDE_MODEL: 覆盖 Claude/Anthropic 模型名（用于 /v1/messages 与 2anthropic）
- TEST_ENABLE_ANTHROPIC_UPSTREAM: 设为 1 时才测试 /anthropic/v1/chat/completions
"""

import asyncio
import ast
import json
import os
import time
from decimal import Decimal, getcontext
from typing import Any, Dict, List, Optional, Tuple

import httpx

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


class SkipTest(Exception):
    pass


def _env_first(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if value:
            return value
    return default


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    s = str(value)
    if len(s) <= 8:
        return "*" * len(s)
    return f"{s[:4]}...{s[-4:]}"


def _server_origin() -> str:
    origin = _env_first("TEST_SERVER_ORIGIN")
    if origin:
        return origin.rstrip("/")
    host = _env_first("SERVER_HOST", default="127.0.0.1") or "127.0.0.1"
    port = _env_first("SERVER_PORT", default="8000") or "8000"
    return f"http://{host}:{port}".rstrip("/")


def _proxy_api_key() -> str:
    key = _env_first("TEST_PROXY_API_KEY")
    if key:
        return key
    keys = _env_first("VALID_API_KEYS", default="") or ""
    first = (keys.split(",")[0] if keys else "").strip()
    return first or "sk-abc123"


def _auth_headers(proxy_key: str) -> Dict[str, str]:
    if not proxy_key:
        return {}
    return {"Authorization": f"Bearer {proxy_key}"}


def _raise_if_openai_error(status_code: int, body: Any) -> None:
    if status_code < 400:
        if isinstance(body, dict):
            err = body.get("error")
            if not isinstance(err, dict):
                return
            if not (err.get("message") or err.get("type") or err.get("code")):
                return

    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        err = body["error"]
        raise RuntimeError(f"HTTP {status_code} | {err.get('type')} | {err.get('code')} | {err.get('message')}")
    raise RuntimeError(f"HTTP {status_code} | {str(body)[:200]}")


def _print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def safe_eval_decimal_expression(expression: str) -> Decimal:
    """
    用 Decimal 安全计算简单数学表达式（仅允许数字、括号、+ - * /）。
    仅用于测试脚本里模拟工具执行，避免直接 eval。
    """
    getcontext().prec = 50
    tree = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            literal = ast.get_source_segment(expression, node)
            return Decimal(literal) if literal is not None else Decimal(str(node.value))

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = _eval(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value

        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right

        raise ValueError("Unsupported expression")

    return _eval(tree)


async def _run_test(name: str, coro) -> Tuple[str, str]:
    try:
        await coro
        return name, "[OK]"
    except SkipTest as e:
        return name, f"[SKIP] {str(e)[:160]}"
    except Exception as e:
        return name, f"[FAIL] {str(e)[:160]}"


async def unit_tests() -> None:
    _print_section("Unit Tests（无需启动服务）")

    import app as opendify_app

    # 1) 工具提取 + OpenAI 响应转换
    tool_payload = {
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "calculate", "arguments": {"expression": "2 + 2"}},
            }
        ]
    }
    text_with_tools = f"hello\n```json\n{json.dumps(tool_payload, ensure_ascii=False)}\n```\n"
    calls = opendify_app.extract_tool_invocations(text_with_tools)
    assert isinstance(calls, list) and calls, "extract_tool_invocations 未提取到 tool_calls"

    cleaned = opendify_app.remove_tool_json_content(text_with_tools)
    assert "tool_calls" not in cleaned, "remove_tool_json_content 未移除 tool_calls code block"

    dify_resp = {
        "message_id": "chatcmpl_unit_1",
        "created_at": int(time.time()),
        "answer": text_with_tools,
        "metadata": {"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}},
    }
    openai_resp, _ = opendify_app.transform_dify_to_openai_response(dify_resp, "UnitModel")
    assert openai_resp["choices"][0]["finish_reason"] == "tool_calls"
    assert openai_resp["choices"][0]["message"].get("tool_calls"), "transform_dify_to_openai_response 未输出 tool_calls"

    # 2) OpenAI Chat -> Anthropic Messages 请求转换
    openai_req = {
        "model": "UnitDisplayModel",
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "calculate", "arguments": "{\"expression\":\"1+1\"}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "name": "calculate", "content": "2"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "math",
                    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
                },
            }
        ],
        "tool_choice": "required",
        "max_tokens": 32,
        "stream": False,
    }
    anthropic_req = opendify_app.transform_openai_chat_to_anthropic_request(
        openai_req,
        upstream_model="claude-unit",
        default_max_tokens=1024,
    )
    assert anthropic_req.get("model") == "claude-unit"
    assert isinstance(anthropic_req.get("messages"), list) and anthropic_req["messages"], "Anthropic messages 为空"
    assert isinstance(anthropic_req.get("tools"), list) and anthropic_req["tools"], "tools 未转换"
    assert (anthropic_req.get("tool_choice") or {}).get("type") == "any", "tool_choice=required 未映射为 any"

    # 3) Anthropic 响应 -> OpenAI ChatCompletion
    anthropic_msg = {
        "id": "msg_unit_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-unit",
        "content": [
            {"type": "text", "text": "ok"},
            {"type": "tool_use", "id": "toolu_1", "name": "calculate", "input": {"expression": "2+2"}},
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    openai_from_anthropic = opendify_app.transform_anthropic_to_openai_chat_completion(
        anthropic_msg,
        request_model="UnitDisplayModel",
    )
    assert openai_from_anthropic["choices"][0]["finish_reason"] == "tool_calls"
    assert openai_from_anthropic["choices"][0]["message"].get("tool_calls"), "Anthropic tool_use 未映射为 tool_calls"

    # 4) SSE parser（Anthropic raw stream）
    class _FakeResponse:
        def __init__(self, lines: List[str]):
            self._lines = lines

        async def aiter_lines(self):
            for line in self._lines:
                yield line

    fake_lines = [
        "event: message_start",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\"}}",
        "",
        "event: content_block_delta",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}",
        "",
        "event: message_stop",
        "data: {\"type\":\"message_stop\"}",
        "",
    ]
    events: List[str] = []
    async for _, ev in opendify_app._iter_anthropic_sse_events(_FakeResponse(fake_lines)):
        events.append(ev.get("type") or "")
    assert events[:1] == ["message_start"]
    assert "message_stop" in events

    # 5) 2anthropic: Claude Messages -> OpenAI ChatCompletions request
    claude_req = {
        "model": "claude-in",
        "max_tokens": 321,
        "system": "you are helpful",
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "tool_use", "id": "toolu_1", "name": "calculate", "input": {"expression": "1+1"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "2"}]},
                ],
            },
        ],
        "tools": [
            {
                "name": "calculate",
                "description": "math",
                "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
            }
        ],
        "tool_choice": {"type": "any"},
        "stop_sequences": ["STOP"],
        "temperature": 0.2,
        "top_p": 0.9,
        "stream": False,
    }
    openai_req = opendify_app.transform_claude_messages_to_openai_chat_completion_request(
        claude_req,
        upstream_model="upstream-model",
        default_max_tokens=1024,
    )
    assert openai_req["model"] == "upstream-model"
    assert openai_req["max_tokens"] == 321
    assert openai_req.get("stop") == ["STOP"]
    assert isinstance(openai_req.get("messages"), list) and openai_req["messages"]
    assert openai_req["messages"][0]["role"] == "system"
    assert sum(1 for m in openai_req["messages"] if m.get("role") == "user") == 1, "tool_result-only user message 不应生成空 user 消息"

    tool_msgs = [m for m in openai_req["messages"] if m.get("role") == "tool"]
    assert tool_msgs, "未生成 role=tool 的消息（tool_result）"
    assert tool_msgs[0].get("tool_call_id") == "toolu_1"
    assert "name" not in tool_msgs[0], "OpenAI tool message 不应包含 name 字段"

    assistant_msgs = [m for m in openai_req["messages"] if m.get("role") == "assistant"]
    assert assistant_msgs and isinstance(assistant_msgs[0].get("tool_calls"), list)
    assert assistant_msgs[0]["tool_calls"][0]["id"] == "toolu_1"

    # 5.1) assistant tool_calls without text -> OpenAI content should be null
    claude_req2 = {
        "model": "claude-in",
        "max_tokens": 1,
        "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_2", "name": "calculate", "input": {"expression": "1+1"}}]},
        ],
        "tools": [
            {"name": "calculate", "description": "math", "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}}},
        ],
        "stream": False,
    }
    openai_req2 = opendify_app.transform_claude_messages_to_openai_chat_completion_request(
        claude_req2,
        upstream_model="upstream-model",
        default_max_tokens=1024,
    )
    assistant2 = [m for m in openai_req2["messages"] if m.get("role") == "assistant"][0]
    assert assistant2.get("tool_calls"), "未生成 tool_calls"
    assert assistant2.get("content") is None, "assistant tool_calls 且无文本时 content 应为 null"

    # 6) 2anthropic: OpenAI ChatCompletions -> Claude Message response
    openai_resp = {
        "id": "chatcmpl_unit_1",
        "object": "chat.completion",
        "created": 0,
        "model": "upstream-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "calculate", "arguments": "{\"expression\":\"2+2\"}"}}
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    claude_msg = opendify_app.transform_openai_chat_completion_to_claude_message(openai_resp, request_model="claude-in")
    assert claude_msg["type"] == "message"
    assert claude_msg["role"] == "assistant"
    assert claude_msg["model"] == "claude-in"
    assert claude_msg["stop_reason"] == "tool_use"
    assert claude_msg["usage"]["input_tokens"] == 10
    assert claude_msg["usage"]["output_tokens"] == 5
    assert any(b.get("type") == "tool_use" for b in claude_msg.get("content") or [])

    # 7) 2anthropic: OpenAI SSE -> Claude SSE events
    async def _collect_sse_payloads(gen) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        async for chunk in gen:
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    payloads.append(json.loads(line[6:]))
        return payloads

    openai_fake_lines = [
        'data: {"id":"chatcmpl_1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
        "",
        'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}',
        "",
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"calculate","arguments":"{\\"expression\\":\\"1"}}]},"finish_reason":null}]}',
        "",
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"+\\"1\\"}"}}]},"finish_reason":null}]}',
        "",
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
        "",
        "data: [DONE]",
        "",
    ]
    payloads = await _collect_sse_payloads(
        opendify_app.stream_openai_chat_completion_as_claude_message(
            _FakeResponse(openai_fake_lines),
            model="claude-in",
            message_id="msg_1",
        )
    )
    types = [p.get("type") for p in payloads if isinstance(p, dict)]
    assert "message_start" in types
    assert "content_block_delta" in types
    assert "message_stop" in types
    tool_starts = [
        p
        for p in payloads
        if p.get("type") == "content_block_start" and (p.get("content_block") or {}).get("type") == "tool_use"
    ]
    assert tool_starts, "stream 未输出 tool_use content_block_start"

    print("Unit Tests: OK")


async def integration_tests() -> None:
    origin = _server_origin()
    proxy_key = _proxy_api_key()

    openai_base = f"{origin}/v1"
    anthropic_origin = f"{origin}/anthropic"
    anthropic_openai_base = f"{anthropic_origin}/v1"

    _print_section("Integration Tests（需要服务已启动）")
    print("server_origin:", origin)
    print("proxy_api_key:", _mask_secret(proxy_key))

    async with httpx.AsyncClient() as client:
        # Reachability check
        try:
            await client.get(f"{openai_base}/models", timeout=3.0)
        except Exception as e:
            raise SkipTest(f"服务不可达：{e}")

        headers = _auth_headers(proxy_key)

        # 0) /v1/models
        _print_section("0: /v1/models + /v1/models/{model_id}")
        r = await client.get(f"{openai_base}/models", headers=headers, timeout=30.0)
        data = r.json()
        _raise_if_openai_error(r.status_code, data)
        assert data.get("object") == "list"
        models = data.get("data") or []
        if not models:
            raise SkipTest("未发现 Dify 模型（请检查 DIFY_API_KEYS / DIFY_API_BASE）")
        model_id = _env_first("TEST_DIFY_MODEL") or (models[0].get("id") if isinstance(models[0], dict) else None)
        if not isinstance(model_id, str) or not model_id.strip():
            raise RuntimeError("无法确定 Dify model_id")
        model_id = model_id.strip()
        print("dify_model:", model_id)

        r = await client.get(f"{openai_base}/models/{model_id}", headers=headers, timeout=30.0)
        data = r.json()
        _raise_if_openai_error(r.status_code, data)
        assert data.get("object") == "model"
        assert data.get("id") == model_id

        # 0.2) /anthropic/v1/models (Anthropic SDK shape)
        _print_section("0.2: /anthropic/v1/models（Anthropic SDK shape）")
        anthropic_headers = {"X-API-Key": proxy_key, "anthropic-version": "2023-06-01"}
        r = await client.get(f"{anthropic_origin}/v1/models", headers=anthropic_headers, timeout=30.0)
        data = r.json()
        _raise_if_openai_error(r.status_code, data)
        assert isinstance(data.get("data"), list)
        assert isinstance(data.get("has_more"), bool)
        assert "first_id" in data
        assert "last_id" in data
        if data["data"]:
            first = data["data"][0]
            assert isinstance(first, dict)
            assert isinstance(first.get("id"), str) and first["id"]
            assert first.get("type") == "model"
            assert isinstance(first.get("display_name"), str)
            assert isinstance(first.get("created_at"), str)

            r = await client.get(f"{anthropic_origin}/v1/models/{first['id']}", headers=anthropic_headers, timeout=30.0)
            m = r.json()
            _raise_if_openai_error(r.status_code, m)
            assert m.get("type") == "model"
            assert m.get("id") == first["id"]

        # 0.3) /anthropic/v1/messages/count_tokens
        _print_section("0.3: /anthropic/v1/messages/count_tokens")
        claude_model = _env_first(
            "TEST_CLAUDE_MODEL",
            "UPSTREAM_OPENAI_MODEL",
            "model",
            default="claude-3-5-sonnet-20241022",
        ) or "claude-3-5-sonnet-20241022"
        r = await client.post(
            f"{anthropic_origin}/v1/messages/count_tokens",
            headers=anthropic_headers,
            json={"model": claude_model, "messages": [{"role": "user", "content": "Count these tokens."}]},
            timeout=30.0,
        )
        data = r.json()
        _raise_if_openai_error(r.status_code, data)
        assert isinstance(data.get("token_count"), int)

        # 1) /v1/chat/completions（blocking）
        _print_section("1: /v1/chat/completions（blocking）")
        r = await client.post(
            f"{openai_base}/chat/completions",
            headers=headers,
            json={"model": model_id, "messages": [{"role": "user", "content": "你好，用一句话介绍自己"}], "stream": False},
            timeout=30.0,
        )
        data = r.json()
        _raise_if_openai_error(r.status_code, data)
        assert data.get("object") == "chat.completion"
        assert (data.get("choices") or [])[0]["message"]["role"] == "assistant"

        # 1.1) /v1/chat/completions（multi-turn）
        _print_section("1.1: /v1/chat/completions（multi-turn）")
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "你是一个数学老师。"},
            {"role": "user", "content": "什么是质数？"},
        ]
        r1 = await client.post(
            f"{openai_base}/chat/completions",
            headers=headers,
            json={"model": model_id, "messages": messages, "stream": False},
            timeout=30.0,
        )
        d1 = r1.json()
        _raise_if_openai_error(r1.status_code, d1)
        assistant_msg_1 = (d1.get("choices") or [])[0]["message"]
        assert assistant_msg_1.get("role") == "assistant"
        messages.append(assistant_msg_1)
        messages.append({"role": "user", "content": "举个例子"})

        r2 = await client.post(
            f"{openai_base}/chat/completions",
            headers=headers,
            json={"model": model_id, "messages": messages, "stream": False},
            timeout=30.0,
        )
        d2 = r2.json()
        _raise_if_openai_error(r2.status_code, d2)
        assistant_msg_2 = (d2.get("choices") or [])[0]["message"]
        assert assistant_msg_2.get("role") == "assistant"

        # 1.2) /v1/chat/completions（tools/tool_calls）
        _print_section("1.2: /v1/chat/completions（tools/tool_calls）")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定城市的天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "城市名称，例如：北京"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "温度单位"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        r = await client.post(
            f"{openai_base}/chat/completions",
            headers=headers,
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "北京今天天气怎么样？"}],
                "tools": tools,
                "stream": False,
            },
            timeout=30.0,
        )
        data = r.json()
        _raise_if_openai_error(r.status_code, data)
        msg = (data.get("choices") or [])[0]["message"]
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            print("WARN: 模型未调用工具（这不一定是代理问题）")
        else:
            assert (data.get("choices") or [])[0].get("finish_reason") == "tool_calls"
            for tc in tool_calls:
                assert isinstance(tc.get("id"), str) and tc["id"]
                assert tc.get("type") == "function"
                fn = tc.get("function") or {}
                assert isinstance(fn.get("name"), str) and fn["name"]
                assert isinstance(fn.get("arguments"), str)
                json.loads(fn["arguments"])

        # 1.3) /v1/chat/completions（multi-turn + tool_calls）
        _print_section("1.3: /v1/chat/completions（multi-turn + tool_calls）")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "执行数学计算",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string", "description": "例如：2 + 2"}},
                        "required": ["expression"],
                    },
                },
            }
        ]
        messages = [{"role": "user", "content": "计算 123.456321 * 456.321"}]
        r1 = await client.post(
            f"{openai_base}/chat/completions",
            headers=headers,
            json={"model": model_id, "messages": messages, "tools": tools, "stream": False},
            timeout=30.0,
        )
        d1 = r1.json()
        _raise_if_openai_error(r1.status_code, d1)
        assistant_msg_1 = (d1.get("choices") or [])[0]["message"]
        messages.append(assistant_msg_1)
        tool_calls = assistant_msg_1.get("tool_calls") or []
        if not tool_calls:
            print("WARN: 模型未调用工具（这不一定是代理问题）")
        else:
            for idx, tc in enumerate(tool_calls):
                call_id = tc.get("id") or f"call_{idx}"
                try:
                    args = json.loads((tc.get("function") or {}).get("arguments") or "{}")
                except Exception:
                    args = {}
                expression = args.get("expression", "")
                tool_result = str(safe_eval_decimal_expression(expression)) if isinstance(expression, str) and expression else ""
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": (tc.get("function") or {}).get("name") or "calculate",
                        "content": tool_result,
                    }
                )

            r2 = await client.post(
                f"{openai_base}/chat/completions",
                headers=headers,
                json={"model": model_id, "messages": messages, "tools": tools, "stream": False},
                timeout=30.0,
            )
            d2 = r2.json()
            _raise_if_openai_error(r2.status_code, d2)
            assistant_msg_2 = (d2.get("choices") or [])[0]["message"]
            assert assistant_msg_2.get("role") == "assistant"

        # 2) /v1/chat/completions（stream + include_usage）
        _print_section("2: /v1/chat/completions（stream + include_usage）")
        role_received = False
        finish_received = False
        usage_received = False
        content_parts: List[str] = []

        async with client.stream(
            "POST",
            f"{openai_base}/chat/completions",
            headers=headers,
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "用一句话介绍 Python"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            timeout=30.0,
        ) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                choices = chunk.get("choices") or []
                if not choices:
                    if isinstance(chunk.get("usage"), dict):
                        usage_received = True
                    continue
                delta = choices[0].get("delta") or {}
                if "role" in delta:
                    role_received = True
                if "content" in delta and delta["content"] is not None:
                    content_parts.append(delta["content"])
                if choices[0].get("finish_reason") is not None:
                    finish_received = True

        assert role_received, "未收到 role"
        assert content_parts, "未收到 content"
        assert finish_received, "未收到 finish_reason"
        assert usage_received, "未收到 include_usage chunk"

        # 3) /v1/responses（blocking）
        _print_section("3: /v1/responses（blocking）")
        r = await client.post(
            f"{openai_base}/responses",
            headers=headers,
            json={"model": model_id, "input": "用一句话介绍 FastAPI", "stream": False},
            timeout=30.0,
        )
        data = r.json()
        _raise_if_openai_error(r.status_code, data)
        assert data.get("object") == "response"
        assert isinstance(data.get("output"), list)

        # 4) /v1/responses（stream）
        _print_section("4: /v1/responses（stream）")
        created_received = False
        completed_received = False
        last_seq: Optional[int] = None
        text_parts = []

        async with client.stream(
            "POST",
            f"{openai_base}/responses",
            headers=headers,
            json={"model": model_id, "input": "用一句话介绍 HTTP", "stream": True},
            timeout=30.0,
        ) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                assert isinstance(event.get("sequence_number"), int), "Responses stream 未返回 sequence_number"
                if last_seq is not None:
                    assert event["sequence_number"] == last_seq + 1, "sequence_number 不连续"
                last_seq = event["sequence_number"]

                if event.get("type") == "response.created":
                    created_received = True
                if event.get("type") == "response.output_text.delta":
                    text_parts.append(event.get("delta") or "")
                if event.get("type") == "response.completed":
                    completed_received = True

        assert created_received, "未收到 response.created"
        assert text_parts, "未收到 output_text.delta"
        assert completed_received, "未收到 response.completed"

        # 4.1) /v1/messages（Claude Messages, Dify 后端）
        _print_section("4.1: /v1/messages（Claude blocking）")
        r = await client.post(
            f"{openai_base}/messages",
            headers=anthropic_headers,
            json={
                "model": model_id,
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "你好，用一句话介绍自己"}],
                "stream": False,
            },
            timeout=30.0,
        )
        data = r.json()
        _raise_if_openai_error(r.status_code, data)
        assert data.get("type") == "message"
        assert isinstance(data.get("id"), str) and data["id"]
        assert data.get("role") == "assistant"
        assert data.get("model") == model_id
        assert isinstance(data.get("content"), list)

        _print_section("4.2: /v1/messages（Claude stream）")
        message_start_received = False
        message_stop_received = False
        text_parts = []
        tool_use_received = False

        async with client.stream(
            "POST",
            f"{openai_base}/messages",
            headers=anthropic_headers,
            json={
                "model": model_id,
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "用一句话介绍 Rust"}],
                "stream": True,
            },
            timeout=30.0,
        ) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if not payload:
                    continue
                event = json.loads(payload)
                if event.get("type") == "message_start":
                    message_start_received = True
                    msg = event.get("message") or {}
                    assert isinstance(msg.get("id"), str) and msg["id"]
                    assert msg.get("role") == "assistant"
                    assert msg.get("model") == model_id
                elif event.get("type") == "content_block_start":
                    content_block = event.get("content_block") or {}
                    if (content_block or {}).get("type") == "tool_use":
                        tool_use_received = True
                elif event.get("type") == "content_block_delta":
                    delta = event.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        text_parts.append(delta.get("text") or "")
                    elif delta.get("type") == "input_json_delta":
                        tool_use_received = True
                elif event.get("type") == "message_stop":
                    message_stop_received = True
                    break

        assert message_start_received, "未收到 message_start"
        assert text_parts or tool_use_received, "未收到 content_block_delta/tool_use"
        assert message_stop_received, "未收到 message_stop"

        upstream_url = _env_first("UPSTREAM_OPENAI_BASE_URL", "UPSTREAM_BASE_URL", "base_url", default="") or ""
        upstream_key = _env_first("UPSTREAM_OPENAI_API_KEY", "UPSTREAM_API_KEY", "api_key", default="") or ""
        if upstream_url and upstream_key:
            # 5) /anthropic/v1/messages（2anthropic blocking + stream）
            _print_section("5: /anthropic/v1/messages（2anthropic blocking）")
            r = await client.post(
                f"{anthropic_origin}/v1/messages",
                headers=anthropic_headers,
                json={
                    "model": claude_model,
                    "max_tokens": 128,
                    "messages": [{"role": "user", "content": "你好，用一句话介绍自己"}],
                    "stream": False,
                },
                timeout=60.0,
            )
            data = r.json()
            _raise_if_openai_error(r.status_code, data)
            assert data.get("type") == "message"
            assert data.get("role") == "assistant"
            assert isinstance(data.get("content"), list)

            _print_section("6: /anthropic/v1/messages（2anthropic stream）")
            message_start_received = False
            message_stop_received = False
            text_parts = []
            tool_use_received = False

            async with client.stream(
                "POST",
                f"{anthropic_origin}/v1/messages",
                headers=anthropic_headers,
                json={
                    "model": claude_model,
                    "max_tokens": 128,
                    "messages": [{"role": "user", "content": "用一句话介绍 Rust"}],
                    "stream": True,
                },
                timeout=60.0,
            ) as r:
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if not payload:
                        continue
                    event = json.loads(payload)
                    if event.get("type") == "message_start":
                        message_start_received = True
                    elif event.get("type") == "content_block_start":
                        content_block = event.get("content_block") or {}
                        if (content_block or {}).get("type") == "tool_use":
                            tool_use_received = True
                    elif event.get("type") == "content_block_delta":
                        delta = event.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            text_parts.append(delta.get("text") or "")
                        elif delta.get("type") == "input_json_delta":
                            tool_use_received = True
                    elif event.get("type") == "message_stop":
                        message_stop_received = True
                        break

            assert message_start_received, "未收到 message_start"
            assert text_parts or tool_use_received, "未收到 content_block_delta/tool_use"
            assert message_stop_received, "未收到 message_stop"
        else:
            print("SKIP: 未配置 UPSTREAM_OPENAI_BASE_URL/api_key（或 base_url/api_key），跳过 /anthropic/v1/messages")

        # 6) /anthropic/v1/models + /anthropic/v1/chat/completions（需配置 ANTHROPIC_API_KEY）
        _print_section("7: /anthropic/v1/models")
        r = await client.get(f"{anthropic_openai_base}/models", headers=headers, timeout=30.0)
        data = r.json()
        _raise_if_openai_error(r.status_code, data)
        assert data.get("object") == "list"
        a_models = data.get("data") or []
        if not a_models:
            print("SKIP: 未配置 ANTHROPIC_MODEL（或 model），/anthropic/v1/models 为空")
        else:
            a_model_id = a_models[0].get("id") if isinstance(a_models[0], dict) else None
            if not isinstance(a_model_id, str) or not a_model_id.strip():
                raise RuntimeError("无法确定 Anthropic model_id")
            a_model_id = a_model_id.strip()
            print("anthropic_model:", a_model_id)

            r = await client.get(f"{anthropic_openai_base}/models/{a_model_id}", headers=headers, timeout=30.0)
            data = r.json()
            _raise_if_openai_error(r.status_code, data)
            assert data.get("object") == "model"
            assert data.get("id") == a_model_id

            enable_anthropic_upstream = (_env_first("TEST_ENABLE_ANTHROPIC_UPSTREAM", default="") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
            anthropic_api_key = _env_first("ANTHROPIC_API_KEY", "ANTHROPIC_KEY", default="") or ""
            if not enable_anthropic_upstream:
                print("SKIP: 未开启 TEST_ENABLE_ANTHROPIC_UPSTREAM，跳过 /anthropic/v1/chat/completions")
            elif not anthropic_api_key:
                print("SKIP: 未配置 ANTHROPIC_API_KEY，跳过 /anthropic/v1/chat/completions")
            else:
                _print_section("8: /anthropic/v1/chat/completions（blocking）")
                r = await client.post(
                    f"{anthropic_openai_base}/chat/completions",
                    headers=headers,
                    json={"model": a_model_id, "messages": [{"role": "user", "content": "你好，用一句话介绍自己"}], "stream": False},
                    timeout=60.0,
                )
                data = r.json()
                _raise_if_openai_error(r.status_code, data)
                assert data.get("object") == "chat.completion"
                assert (data.get("choices") or [])[0]["message"]["role"] == "assistant"

                _print_section("9: /anthropic/v1/chat/completions（stream + include_usage）")
                role_received = False
                finish_received = False
                usage_received = False
                content_parts = []

                async with client.stream(
                    "POST",
                    f"{anthropic_openai_base}/chat/completions",
                    headers=headers,
                    json={
                        "model": a_model_id,
                        "messages": [{"role": "user", "content": "用一句话介绍 FastAPI"}],
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                    timeout=60.0,
                ) as r:
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break
                        chunk = json.loads(payload)
                        choices = chunk.get("choices") or []
                        if not choices:
                            if isinstance(chunk.get("usage"), dict):
                                usage_received = True
                            continue
                        delta = choices[0].get("delta") or {}
                        if "role" in delta:
                            role_received = True
                        if "content" in delta and delta["content"] is not None:
                            content_parts.append(delta["content"])
                        if choices[0].get("finish_reason") is not None:
                            finish_received = True

                assert role_received, "未收到 role"
                assert content_parts, "未收到 content"
                assert finish_received, "未收到 finish_reason"
                assert usage_received, "未收到 include_usage chunk"

    print("Integration Tests: OK")


async def main() -> None:
    results: List[Tuple[str, str]] = []

    for name, coro in [
        ("Unit", unit_tests()),
        ("Integration", integration_tests()),
    ]:
        results.append(await _run_test(name, coro))
        await asyncio.sleep(0.3)

    _print_section("测试结果摘要")
    for n, r in results:
        print(f"{n:12} {r}")


if __name__ == "__main__":
    asyncio.run(main())
