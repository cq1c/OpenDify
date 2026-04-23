"""Dify → OpenAI 响应转换 + SSE 字节流拆分。"""

import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from .config import CONVERSATION_MODE, PROMPT_DIALECT
from .dialects import get_dialect
from .sessions import sessions
from .utils import fast_id

extract_tool_calls = get_dialect(PROMPT_DIALECT).extract_tool_calls


def build_openai_response(
    dify_resp: Dict[str, Any],
    model: str,
    tool_token: Optional[str] = None,
    session: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    answer = dify_resp.get("answer", "")
    tool_calls: Optional[List[Dict[str, Any]]] = None
    if answer:
        answer, tool_calls = extract_tool_calls(answer, tool_token, tools=tools)
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": answer.strip() if answer and answer.strip() else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage_raw = (
        (dify_resp.get("metadata") or {}).get("usage")
        if isinstance(dify_resp.get("metadata"), dict)
        else {}
    )
    if not isinstance(usage_raw, dict):
        usage_raw = {}

    # conversation 模式下返回累积 usage，便于客户端正确显示任务级上下文占用。
    if session is not None and CONVERSATION_MODE == "auto":
        usage_out = sessions.accumulate_usage(session, usage_raw)
    else:
        usage_out = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens") or 0),
            "completion_tokens": int(usage_raw.get("completion_tokens") or 0),
            "total_tokens": int(usage_raw.get("total_tokens") or 0),
        }

    resp = {
        "id": f"chatcmpl-{fast_id()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": "fp_dify",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "logprobs": None,
            }
        ],
        "usage": usage_out,
    }
    return resp, dify_resp.get("conversation_id")


async def iter_dify_sse(rsp: httpx.Response) -> AsyncGenerator[Dict[str, Any], None]:
    buf = bytearray()
    async for chunk in rsp.aiter_bytes(4096):
        buf.extend(chunk)
        while b"\n" in buf:
            idx = buf.index(b"\n")
            line = bytes(buf[:idx]).strip()
            buf = buf[idx + 1 :]
            if not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if not payload:
                continue
            try:
                data = json.loads(payload)
            except Exception:
                continue
            if isinstance(data, dict):
                yield data
