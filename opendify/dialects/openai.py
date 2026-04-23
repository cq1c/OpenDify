"""
OpenAI / Hermes / Qwen 方言。

Query 文本 (ChatML):
  <|im_start|>system
  {system + tools 说明}
  <|im_end|>
  <|im_start|>user
  {user content}
  <|im_end|>
  <|im_start|>assistant
  {assistant text}
  <tool_call>
  {"name": "X", "arguments": {...}}
  </tool_call>
  <|im_end|>
  <|im_start|>tool
  {"tool_call_id": "call_xxx", "name": "X", "content": "..."}
  <|im_end|>

期望模型输出:
  <tool_call [token="XXX"]>
  {"name": "X", "arguments": {...}}
  </tool_call>
  (可连续多个 <tool_call> 表示并发)
"""

import json
import re
import secrets
from typing import Any, Dict, List, Optional, Tuple

from ..config import (
    CONVERSATION_MODE,
    ONLY_RECENT_MESSAGES,
    SIMPLIFIED_TOOL_DEFS,
    STRIP_SYSTEM_AFTER_FIRST,
    SYSTEM_PROMPT_MAX_LENGTH,
    TOOL_CALL_STRICTNESS,
    TOOL_DESC_MAX_LENGTH,
    USE_TOOL_TOKEN,
    logger,
)
from ..tool_calls import (
    _coerce_tool_calls_parsed,
    _normalize_tool_calls,
    _robust_json_parse,
)
from ..tool_digest import tool_desc_digest
from ..utils import extract_text, truncate

CLOSE_TAG = "</tool_call>"
_OPEN_ATTR = r'(?:\s+token="([^"]*)")?'
OPEN_TAG_PATTERN = re.compile(rf"<tool_call{_OPEN_ATTR}\s*>", re.IGNORECASE)
_BLOCK_PATTERN = re.compile(
    rf"<tool_call{_OPEN_ATTR}\s*>(.*?)</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
HOLDBACK = 40  # `<tool_call token="XXXXXX">` 约 28 字, 预留余量


# ────────────────────────────────────────
#  渲染
# ────────────────────────────────────────


def _build_open_tag(token: Optional[str]) -> str:
    if token and USE_TOOL_TOKEN:
        return f'<tool_call token="{token}">'
    return "<tool_call>"


def _im(role: str, body: str) -> str:
    return f"<|im_start|>{role}\n{body}\n<|im_end|>"


def _render_tool_def(func: Dict[str, Any], dify_key: Optional[str]) -> str:
    name = func.get("name", "unknown")
    raw_desc = func.get("description", "") or ""
    cached = tool_desc_digest.load(name, raw_desc) if raw_desc else None
    if cached:
        desc = cached
    else:
        desc = truncate(raw_desc, TOOL_DESC_MAX_LENGTH * 2)
        if dify_key and raw_desc:
            tool_desc_digest.schedule_generate(dify_key, name, raw_desc)
    params = func.get("parameters") or {}
    separators = (",", ":") if SIMPLIFIED_TOOL_DEFS else (", ", ": ")
    obj = {
        "type": "function",
        "function": {"name": name, "description": desc, "parameters": params},
    }
    return json.dumps(obj, ensure_ascii=False, separators=separators)


def _build_system_block(
    system_texts: List[str],
    tools: List[Dict[str, Any]],
    token: Optional[str],
    prev_tokens: Optional[List[str]],
    dify_key: Optional[str],
    tool_choice: Any,
    level: int,
) -> str:
    parts: List[str] = []
    if system_texts:
        parts.append("\n\n".join(system_texts))

    if tools:
        tool_lines: List[str] = []
        for t in tools:
            if not isinstance(t, dict) or t.get("type") != "function":
                continue
            fn = t.get("function") or {}
            if isinstance(fn, dict) and fn.get("name"):
                tool_lines.append(_render_tool_def(fn, dify_key))
        tools_block = "<tools>\n" + "\n".join(tool_lines) + "\n</tools>"
        tag_open = _build_open_tag(token)

        example = (
            f"{tag_open}\n"
            '{"name": "example_tool", "arguments": {"p1": "value1"}}\n'
            "</tool_call>"
        )

        instructions: List[str]
        if level <= 0:
            instructions = [
                "你可以调用以下函数。若需调用, 在正文后追加 <tool_call>{JSON}</tool_call>。"
            ]
        else:
            instructions = [
                "你可以调用以下函数。返回函数签名放在 <tools></tools> 之间。",
                "如需调用, 每个函数调用输出一个 <tool_call> 块, 内容是 JSON:",
                '{"name": "<函数名>", "arguments": <参数对象>}',
                "并发调用时输出多个连续的 <tool_call> 块。",
            ]
        if level >= 2:
            instructions.append(
                f"⚠️ 强制: 工具调用必须包裹在 `{tag_open}` 与 `</tool_call>` 之间, 缺失将被拒收。"
            )
        if level >= 3:
            instructions.append(
                "⚠️ 没有 `</tool_call>` 结尾的响应会被系统丢弃, 用户看不到。"
            )

        if token and USE_TOOL_TOKEN:
            tline = f'本次令牌必须是 "{token}", 严禁自行编造或照抄历史。'
            if prev_tokens:
                expired = ", ".join(f'"{t}"' for t in prev_tokens[-5:])
                tline += f" (历史令牌 {expired} 已全部作废)"
            instructions.append(tline)

        if tool_choice == "required":
            instructions.append("本轮必须调用至少一个工具。")
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            fname = (tool_choice.get("function") or {}).get("name")
            if fname:
                instructions.append(f"请使用 {fname} 工具。")

        parts.append(tools_block)
        parts.append("\n".join(instructions))
        parts.append(f"示例:\n{example}")

    return _im("system", "\n\n".join(parts)) if parts else ""


def _render_assistant(msg: Dict[str, Any]) -> str:
    content = extract_text(msg.get("content"))
    tc = msg.get("tool_calls")
    body_parts: List[str] = []
    if content:
        body_parts.append(content)
    if tc and isinstance(tc, list):
        for call in tc:
            fn = call.get("function") or {}
            name = fn.get("name", "unknown")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            obj = {"name": name, "arguments": args if args is not None else {}}
            body_parts.append(
                "<tool_call>\n"
                + json.dumps(obj, ensure_ascii=False)
                + "\n</tool_call>"
            )
    return _im("assistant", "\n".join(body_parts) if body_parts else "")


def _render_tool_result(msg: Dict[str, Any]) -> str:
    content = extract_text(msg.get("content"))
    name = msg.get("name", "tool")
    call_id = msg.get("tool_call_id", "")
    obj = {"tool_call_id": call_id, "name": name, "content": content}
    return _im("tool", json.dumps(obj, ensure_ascii=False))


def render_query(
    openai_req: Dict[str, Any],
    session: Dict[str, Any],
    dify_key: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    messages: List[Dict[str, Any]] = openai_req.get("messages") or []
    if not messages:
        return None, None

    tools = openai_req.get("tools") or []
    tool_choice = openai_req.get("tool_choice", "auto")
    token = session["token"]
    conversation_id = session.get("conversation_id")
    is_continuation = session["msg_count"] > 1 and conversation_id is not None

    if is_continuation and CONVERSATION_MODE == "auto":
        last_assistant_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if (messages[idx] or {}).get("role") == "assistant":
                last_assistant_idx = idx
                break
        system_msgs = [m for m in messages if (m or {}).get("role") == "system"]
        delta_msgs = (
            messages[last_assistant_idx + 1 :] if last_assistant_idx >= 0 else messages
        )
        selected = system_msgs + [
            m for m in delta_msgs if (m or {}).get("role") != "system"
        ]
    elif ONLY_RECENT_MESSAGES > 0:
        system_msgs = [m for m in messages if (m or {}).get("role") == "system"]
        non_system = [m for m in messages if (m or {}).get("role") != "system"]
        selected = system_msgs + non_system[-ONLY_RECENT_MESSAGES:]
    else:
        selected = messages

    system_texts: List[str] = []
    blocks: List[str] = []
    for msg in selected:
        role = msg.get("role", "")
        if role == "system":
            if STRIP_SYSTEM_AFTER_FIRST and is_continuation:
                system_texts.append("(参照之前的系统指令)")
                continue
            content = extract_text(msg.get("content"))
            if SYSTEM_PROMPT_MAX_LENGTH > 0:
                content = truncate(content, SYSTEM_PROMPT_MAX_LENGTH)
            system_texts.append(content)
        elif role == "user":
            blocks.append(_im("user", extract_text(msg.get("content"))))
        elif role == "assistant":
            blocks.append(_render_assistant(msg))
        elif role == "tool":
            blocks.append(_render_tool_result(msg))

    tool_token: Optional[str] = None
    use_tools = bool(tools) and tool_choice != "none"
    if use_tools:
        tool_token = token

    system_block = _build_system_block(
        system_texts,
        tools if use_tools else [],
        token if use_tools else None,
        session.get("prev_tokens") if use_tools else None,
        dify_key,
        tool_choice if use_tools else "none",
        TOOL_CALL_STRICTNESS,
    )

    out: List[str] = []
    if system_block:
        out.append(system_block)
    out.extend(blocks)

    if use_tools:
        last_non_system = None
        for m in reversed(selected):
            if (m or {}).get("role") != "system":
                last_non_system = m
                break
        if (
            last_non_system
            and last_non_system.get("role") == "tool"
            and TOOL_CALL_STRICTNESS >= 2
        ):
            out.append(
                "[注意]: 上一步刚用过工具, 这一步多半要继续调用。"
                "请以 <tool_call>{JSON}</tool_call> 格式输出。"
            )

    # 约定模型续写 assistant turn
    out.append("<|im_start|>assistant")
    return "\n".join(out), tool_token


# ────────────────────────────────────────
#  解析
# ────────────────────────────────────────


def extract_tool_calls(
    text: str,
    token: Optional[str] = None,
    prev_tokens: Optional[List[str]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """
    从模型输出提取工具调用。分级兜底:
      1. 完整 <tool_call>...</tool_call> 块 (可多个, 支持并发)
      2. 只有开始标签 → 取后续整段当 JSON
      3. 裸 JSON dict {"name":..., "arguments":{}} 扫描 (末尾)
      4. 跨方言兜底: 尝试 generic <tool-calls>[JSON] 和 <function_calls><invoke> 格式
    """
    if not text:
        return text or "", None

    raw_calls: List[Dict[str, Any]] = []
    first_start: Optional[int] = None

    # ── 1) 完整块 (可多个) ──
    for m in _BLOCK_PATTERN.finditer(text):
        matched_token = m.group(1)
        if USE_TOOL_TOKEN and token and matched_token and matched_token != token:
            if not prev_tokens or matched_token not in prev_tokens:
                logger.debug(
                    "tool_call token 不匹配 (got=%s expected=%s), 宽松接受",
                    matched_token,
                    token,
                )
        body = m.group(2).strip()
        parsed = _robust_json_parse(body)
        if isinstance(parsed, dict) and parsed.get("name"):
            raw_calls.append(parsed)
            if first_start is None:
                first_start = m.start()
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and item.get("name"):
                    raw_calls.append(item)
            if first_start is None:
                first_start = m.start()

    if raw_calls:
        return _finalize(text, first_start, raw_calls, tools)

    # ── 2) 只有开始标签, 缺收尾 ──
    opens = list(OPEN_TAG_PATTERN.finditer(text))
    if opens:
        last_open = opens[-1]
        tail = text[last_open.end() :].strip()
        parsed = _robust_json_parse(tail)
        if isinstance(parsed, dict) and parsed.get("name"):
            return _finalize(text, last_open.start(), [parsed], tools)
        if isinstance(parsed, list):
            items = [i for i in parsed if isinstance(i, dict) and i.get("name")]
            if items:
                return _finalize(text, last_open.start(), items, tools)

    # ── 3) 裸 JSON dict 末尾扫描 (无任何 tool_call 标签) ──
    last_brace = text.rfind("}")
    if last_brace >= 0:
        # 向前找到最近的 '{' (平衡括号)
        depth = 0
        start_brace = None
        for i in range(last_brace, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
                if depth == 0:
                    start_brace = i
                    break
        if start_brace is not None:
            parsed = _robust_json_parse(text[start_brace : last_brace + 1])
            if isinstance(parsed, dict) and parsed.get("name") and parsed.get("arguments") is not None:
                return _finalize(text, start_brace, [parsed], tools)

    # ── 4) 跨方言兜底: generic <tool-calls>[JSON] 或 claude <invoke> ──
    from ..tool_calls import extract_tool_calls as _generic_extract
    clean, calls = _generic_extract(text, token, prev_tokens, tools)
    if calls:
        logger.debug("openai dialect: 跨方言兜底 (generic parser) 命中")
        return clean, calls

    return text, None


def _finalize(
    text: str,
    first_start: Optional[int],
    raw_calls: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    shaped = [
        {
            "id": f"call_{secrets.token_hex(8)}",
            "type": "function",
            "function": {
                "name": c.get("name"),
                "arguments": c.get("arguments") if c.get("arguments") is not None else {},
            },
        }
        for c in raw_calls
    ]
    shaped = _coerce_tool_calls_parsed(shaped, tools)
    final = _normalize_tool_calls(shaped)
    clean_text = text[:first_start].rstrip() if first_start is not None else text
    return clean_text, (final or None)
