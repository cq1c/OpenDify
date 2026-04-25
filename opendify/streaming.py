"""Dify SSE → OpenAI SSE 转换（支持工具调用捕获、usage 累计、流量日志）。"""

import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from .config import AUTO_USAGE_MODE, CONVERSATION_MODE, PROMPT_DIALECT, logger
from .dialects import get_dialect
from .responses import iter_dify_sse
from .sessions import sessions
from .traffic_log import traffic_log

_dialect = get_dialect(PROMPT_DIALECT)
TOOL_OPEN_TAG_PATTERN = _dialect.OPEN_TAG_PATTERN
extract_tool_calls = _dialect.extract_tool_calls
_DIALECT_HOLDBACK = getattr(_dialect, "HOLDBACK", 40)


async def stream_and_capture_cid(
    rsp: httpx.Response,
    *,
    model: str,
    message_id: str,
    tool_token: Optional[str],
    include_usage: bool,
    session: Dict[str, Any],
    request_id: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    local_prompt_tokens: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    # 工具调用开始标签可能是：<tool-calls>、<tool-calls token="xxxx">
    # HOLDBACK 取最长可能的开始标签长度上限，保证流里不会提前暴露半截标签。
    detect_tool_tags = tool_token is not None
    HOLDBACK = _DIALECT_HOLDBACK if detect_tool_tags else 0

    accumulated = ""
    sent_up_to = 0
    tool_mode = False
    cid_captured = False
    captured_cid: Optional[str] = None
    final_tool_calls: Optional[List[Dict[str, Any]]] = None
    final_finish_reason = "stop"
    usage_obj: Dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def _chunk(
        delta_content: Optional[str] = None,
        delta_role: Optional[str] = None,
        delta_tool_calls: Optional[List[Dict]] = None,
        finish_reason: Optional[str] = None,
    ) -> str:
        c: Dict[str, Any] = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "system_fingerprint": "fp_dify",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }
        d = c["choices"][0]["delta"]
        if delta_role:
            d["role"] = delta_role
        if delta_content is not None:
            d["content"] = delta_content
        if delta_tool_calls is not None:
            d["tool_calls"] = delta_tool_calls
        return f"data: {json.dumps(c, ensure_ascii=False)}\n\n"

    try:
        yield _chunk(delta_role="assistant")

        async for data in iter_dify_sse(rsp):
            if not cid_captured:
                cid = data.get("conversation_id")
                if isinstance(cid, str) and cid.strip():
                    sessions.update_conversation_id(session, cid)
                    captured_cid = cid
                    cid_captured = True

            event = data.get("event")

            if event in ("message", "agent_message"):
                delta = data.get("answer", "")
                if not delta:
                    continue
                if not isinstance(delta, str):
                    delta = str(delta)
                accumulated += delta

                if tool_mode:
                    continue

                if detect_tool_tags:
                    open_match = TOOL_OPEN_TAG_PATTERN.search(
                        accumulated, sent_up_to
                    )
                    if open_match:
                        tool_mode = True
                        tag_pos = open_match.start()
                        unsent = accumulated[sent_up_to:tag_pos].rstrip()
                        if unsent:
                            yield _chunk(delta_content=unsent)
                        sent_up_to = len(accumulated)
                        continue

                safe_end = (
                    (len(accumulated) - HOLDBACK) if HOLDBACK > 0 else len(accumulated)
                )
                if safe_end > sent_up_to:
                    yield _chunk(delta_content=accumulated[sent_up_to:safe_end])
                    sent_up_to = safe_end

            elif event == "message_end":
                meta_usage = (
                    (data.get("metadata") or {}).get("usage")
                    if isinstance(data.get("metadata"), dict)
                    else None
                )
                if isinstance(meta_usage, dict):
                    if CONVERSATION_MODE == "auto":
                        if AUTO_USAGE_MODE == "passthrough":
                            usage_obj = {
                                "prompt_tokens": int(
                                    meta_usage.get("prompt_tokens") or 0
                                ),
                                "completion_tokens": int(
                                    meta_usage.get("completion_tokens") or 0
                                ),
                                "total_tokens": int(
                                    meta_usage.get("total_tokens") or 0
                                ),
                            }
                        else:
                            usage_obj = sessions.accumulate_usage(
                                session, meta_usage
                            )
                    else:
                        # 非 conversation 模式: 上游每轮独立, prompt_tokens 容易随
                        # Dify 内部模板抖动。优先用本地基于 query 文本计算的稳定值,
                        # 仅当本地估算缺失或上游显著更高 (说明上游算上了模板开销) 时
                        # 退回上游报数。
                        upstream_prompt = int(meta_usage.get("prompt_tokens") or 0)
                        completion = int(meta_usage.get("completion_tokens") or 0)
                        if local_prompt_tokens is not None:
                            prompt = max(local_prompt_tokens, upstream_prompt)
                        else:
                            prompt = upstream_prompt
                        usage_obj = {
                            "prompt_tokens": prompt,
                            "completion_tokens": completion,
                            "total_tokens": prompt + completion,
                        }
                elif local_prompt_tokens is not None and CONVERSATION_MODE != "auto":
                    # 上游没给 usage, 用本地估算保底
                    usage_obj = {
                        "prompt_tokens": local_prompt_tokens,
                        "completion_tokens": 0,
                        "total_tokens": local_prompt_tokens,
                    }

                if tool_mode:
                    _, tool_calls = extract_tool_calls(accumulated, tool_token, tools=tools)
                    if tool_calls:
                        final_tool_calls = tool_calls
                        final_finish_reason = "tool_calls"
                        tc_list = [
                            {
                                "index": idx,
                                "id": tc["id"],
                                "type": "function",
                                "function": tc["function"],
                            }
                            for idx, tc in enumerate(tool_calls)
                        ]
                        yield _chunk(delta_tool_calls=tc_list)
                        yield _chunk(finish_reason="tool_calls")
                    else:
                        rest = accumulated[sent_up_to:]
                        if rest:
                            yield _chunk(delta_content=rest)
                        yield _chunk(finish_reason="stop")
                else:
                    rest = accumulated[sent_up_to:]
                    if rest:
                        yield _chunk(delta_content=rest)
                    yield _chunk(finish_reason="stop")

                if include_usage:
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": message_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "system_fingerprint": "fp_dify",
                                "choices": [],
                                "usage": usage_obj,
                            },
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )

                # 记录 Dify 流式响应摘要
                traffic_log.log_dify_response(
                    request_id,
                    200,
                    {"answer": accumulated, "usage": usage_obj},
                    conversation_id=captured_cid,
                    is_stream_summary=True,
                )

                # 记录最终发送给客户端的结果
                clean_text, _ = (
                    extract_tool_calls(accumulated, tool_token, tools=tools)
                    if tool_token
                    else (accumulated, None)
                )
                traffic_log.log_stream_complete(
                    request_id,
                    accumulated_text=clean_text,
                    tool_calls=final_tool_calls,
                    conversation_id=captured_cid,
                    finish_reason=final_finish_reason,
                )

                yield "data: [DONE]\n\n"
                return

        rest = accumulated[sent_up_to:]
        if rest and not tool_mode:
            yield _chunk(delta_content=rest)
        yield _chunk(finish_reason="stop")

        traffic_log.log_stream_complete(
            request_id,
            accumulated_text=accumulated,
            tool_calls=None,
            conversation_id=captured_cid,
            finish_reason="stop",
        )

        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"流处理错误: {e}")
        traffic_log.log_error(request_id, "stream_processing", str(e))
        yield _chunk(finish_reason="stop")
        yield "data: [DONE]\n\n"
