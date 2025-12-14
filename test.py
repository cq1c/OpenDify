#!/usr/bin/env python3
"""
OpenDify - OpenAI Chat Completions 兼容性测试

运行前：
1) 启动 OpenDify（python app.py）
2) 配置 BASE_URL / API_KEY / MODEL
"""

import asyncio
import ast
import json
from decimal import Decimal, getcontext
from typing import Any, Dict, List, Optional, Tuple

import httpx

# 配置
BASE_URL = "http://127.0.0.1:8000/v1"
API_KEY = "sk-abc123"  # 修改为你的 API key（对应 VALID_API_KEYS）
MODEL = "ChatCoder"  # 修改为 /v1/models 返回的模型名称（Dify 应用名）


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def raise_if_openai_error(status_code: int, body: Dict[str, Any]) -> None:
    if "error" not in body:
        return
    err = body["error"] or {}
    raise RuntimeError(
        f"HTTP {status_code} | {err.get('type')} | {err.get('code')} | {err.get('message')}"
    )


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


async def test_list_models() -> None:
    print_section("测试0: /v1/models 模型列表")
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{BASE_URL}/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=30.0,
        )
        data = r.json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
        raise_if_openai_error(r.status_code, data)

        assert data.get("object") == "list"
        assert isinstance(data.get("data"), list)

        if not data["data"]:
            print("WARN: 未发现可用模型，请检查 DIFY_API_KEYS / DIFY_API_BASE 配置")


async def test_get_model() -> None:
    print_section("测试0.1: /v1/models/{model_id} 模型详情")
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{BASE_URL}/models/{MODEL}",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=30.0,
        )
        data = r.json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
        raise_if_openai_error(r.status_code, data)

        assert data.get("object") == "model"
        assert data.get("id") == MODEL


async def test_basic_chat() -> None:
    print_section("测试1: 基础对话（无工具）")
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "你好，介绍一下你自己"},
    ]

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": MODEL, "messages": messages, "stream": False},
            timeout=30.0,
        )
        data = r.json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
        raise_if_openai_error(r.status_code, data)

        assert data.get("object") == "chat.completion"
        assert "choices" in data and isinstance(data["choices"], list) and data["choices"]
        msg = data["choices"][0]["message"]
        assert msg["role"] == "assistant"
        assert "content" in msg


async def test_multi_turn_conversation() -> None:
    print_section("测试2: 多轮对话（客户端维护 messages）")
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "你是一个数学老师。"},
        {"role": "user", "content": "什么是质数？"},
    ]

    async with httpx.AsyncClient() as client:
        r1 = await client.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": MODEL, "messages": messages, "stream": False},
            timeout=30.0,
        )
        d1 = r1.json()
        raise_if_openai_error(r1.status_code, d1)
        assistant_msg_1 = d1["choices"][0]["message"]
        print("assistant#1:", (assistant_msg_1.get("content") or "")[:100])

        messages.append(assistant_msg_1)
        messages.append({"role": "user", "content": "举个例子"})

        r2 = await client.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": MODEL, "messages": messages, "stream": False},
            timeout=30.0,
        )
        d2 = r2.json()
        raise_if_openai_error(r2.status_code, d2)
        assistant_msg_2 = d2["choices"][0]["message"]
        print("assistant#2:", (assistant_msg_2.get("content") or "")[:100])


async def test_tool_calls() -> None:
    print_section("测试3: tools/tool_calls")
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
    messages = [{"role": "user", "content": "北京今天天气怎么样？"}]

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": MODEL, "messages": messages, "tools": tools, "stream": False},
            timeout=30.0,
        )
        data = r.json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
        raise_if_openai_error(r.status_code, data)

        msg = data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            print("WARN: 模型未调用工具（这不一定是代理问题）")
            return

        assert data["choices"][0]["finish_reason"] == "tool_calls"
        for tc in tool_calls:
            assert isinstance(tc.get("id"), str) and tc["id"]
            assert tc.get("type") == "function"
            fn = tc.get("function") or {}
            assert isinstance(fn.get("name"), str) and fn["name"]
            assert isinstance(fn.get("arguments"), str)
            json.loads(fn["arguments"])


async def test_tool_calls_multi_turn() -> None:
    print_section("测试4: 多轮 + tool_calls（完整流程）")
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
    messages: List[Dict[str, Any]] = [{"role": "user", "content": "计算 123.456321 * 456.321"}]

    async with httpx.AsyncClient() as client:
        r1 = await client.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": MODEL, "messages": messages, "tools": tools, "stream": False},
            timeout=30.0,
        )
        d1 = r1.json()
        raise_if_openai_error(r1.status_code, d1)
        assistant_msg_1 = d1["choices"][0]["message"]
        print(json.dumps(assistant_msg_1, indent=2, ensure_ascii=False))
        messages.append(assistant_msg_1)

        tool_calls = assistant_msg_1.get("tool_calls") or []
        if not tool_calls:
            print("WARN: 模型未调用工具（这不一定是代理问题）")
            return

        for tc in tool_calls:
            args = json.loads(tc["function"]["arguments"])
            expression = args.get("expression", "")
            tool_result = str(safe_eval_decimal_expression(expression))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "content": tool_result,
                }
            )

        r2 = await client.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": MODEL, "messages": messages, "tools": tools, "stream": False},
            timeout=30.0,
        )
        d2 = r2.json()
        raise_if_openai_error(r2.status_code, d2)
        assistant_msg_2 = d2["choices"][0]["message"]
        print("assistant:", assistant_msg_2.get("content", ""))


async def test_streaming() -> None:
    print_section("测试5: 流式（SSE）")
    messages = [{"role": "user", "content": "用一句话介绍 Python"}]

    role_received = False
    finish_reason_received = False
    content_parts: List[str] = []

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": MODEL, "messages": messages, "stream": True},
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
                    continue

                delta = choices[0].get("delta") or {}
                if "role" in delta:
                    role_received = True
                if "content" in delta and delta["content"] is not None:
                    content_parts.append(delta["content"])
                if choices[0].get("finish_reason") is not None:
                    finish_reason_received = True

    assert role_received, "未收到 role"
    assert content_parts, "未收到 content"
    assert finish_reason_received, "未收到 finish_reason"
    print("content_len:", len("".join(content_parts)))


async def run_test(name: str, coro) -> Tuple[str, str]:
    try:
        await coro
        return name, "[OK]"
    except Exception as e:
        return name, f"[FAIL] {str(e)[:120]}"


async def main() -> None:
    print_section("OpenDify - OpenAI 兼容性测试")
    results: List[Tuple[str, str]] = []

    for name, coro in [
        ("模型列表", test_list_models()),
        ("模型详情", test_get_model()),
        ("基础对话", test_basic_chat()),
        ("多轮对话", test_multi_turn_conversation()),
        ("工具调用", test_tool_calls()),
        ("多轮+工具", test_tool_calls_multi_turn()),
        ("流式响应", test_streaming()),
    ]:
        results.append(await run_test(name, coro))
        await asyncio.sleep(1)

        if name == "基础对话" and results[-1][1].startswith("[FAIL]"):
            print("基础功能不可用，后续测试将继续运行但可能大量失败。")

    print_section("测试结果摘要")
    for n, r in results:
        print(f"{n:10} {r}")


if __name__ == "__main__":
    asyncio.run(main())
