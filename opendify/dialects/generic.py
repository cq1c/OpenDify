"""
Generic 方言: `[Role]: content` 的纯文本 + `<tool-calls>[JSON]</tool-calls>`。
这是 OpenDify 的原始默认行为, 所有逻辑都从 tool_prompt / tool_calls 复用。
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from ..config import (
    CONVERSATION_MODE,
    ONLY_RECENT_MESSAGES,
    STRIP_SYSTEM_AFTER_FIRST,
    SYSTEM_PROMPT_MAX_LENGTH,
    TOOL_CALL_STRICTNESS,
)
from ..tool_calls import TOOL_OPEN_TAG_PATTERN, extract_tool_calls as _extract
from ..tool_prompt import TOOL_CLOSE_TAG, build_open_tag, generate_tool_prompt
from ..utils import extract_text, truncate

OPEN_TAG_PATTERN = TOOL_OPEN_TAG_PATTERN
CLOSE_TAG = TOOL_CLOSE_TAG
HOLDBACK = 40
extract_tool_calls = _extract


def _build_front_reminder(level: int, tag_open: str, tag_close: str) -> str:
    if level <= 1:
        return ""
    if level == 2:
        return f"⚠️ 强制规则：涉及工具调用时，必须以 `{tag_open}` 开头、`{tag_close}` 闭合。"
    if level == 3:
        return f"🚫 严重警告：违反格式 = 输出丢弃。必须以 `{tag_open}` 起、`{tag_close}` 止。"
    if level == 4:
        return (
            "☢️ 最后一次机会 ☢️\n"
            f"规则一：开头必须是 `{tag_open}`\n"
            f"规则二：结尾必须是 `{tag_close}`\n"
            f"违反任意一条 = 本轮作废。"
        )
    if level == 5:
        return (
            f"⚠️ 傻逼模型听好：敢不以 `{tag_open}` 开头、`{tag_close}` 结尾，\n"
            f"老子直接把你输出丢垃圾桶，用户一个字都看不到。"
        )
    if level == 6:
        return f"夫作答之时，必以 `{tag_open}` 启，以 `{tag_close}` 终，工具之列书其中。"
    return ""


def _build_tool_followup_reminder(level: int, tag_open: str, tag_close: str) -> str:
    if level <= 1:
        return ""
    if level == 2:
        return f"[注意]: 上一步你刚用过工具 → 这一步大概率还要继续调用。请按 `{tag_open}...{tag_close}` 格式输出。"
    if level == 3:
        return (
            f"[重要警告]: 上一步刚用过工具 → 本轮十有八九还要用。\n"
            f"不写 `{tag_open}` 标签 = 本次输出作废。"
        )
    if level == 4:
        return (
            "☢️ [铁律再确认]: 上一步用了工具，这一步继续用的概率极高。\n"
            f"必须 `{tag_open}` 起、`{tag_close}` 止。偏离即废稿。"
        )
    if level == 5:
        return (
            "[给老子听好]: 上一步你刚用过工具，这一步妈逼还得接着用。\n"
            f"别装死不写标签 —— 那是你智商低下、脑残透顶的表现。\n"
            f"`{tag_open}` 起、`{tag_close}` 终，立刻！马上！"
        )
    if level == 6:
        return f"[谨识之]：前番既已用器，此番多半续用。请守 `{tag_open}` ... `{tag_close}` 之式，依前例而书。"
    return ""


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

    query_parts: List[str] = []
    for msg in selected:
        role = msg.get("role", "")
        content = extract_text(msg.get("content"))
        if role == "system":
            if STRIP_SYSTEM_AFTER_FIRST and is_continuation:
                query_parts.append("[System]: (参照之前的系统指令)")
                continue
            if SYSTEM_PROMPT_MAX_LENGTH > 0:
                content = truncate(content, SYSTEM_PROMPT_MAX_LENGTH)
            query_parts.append(f"[System]: {content}")
        elif role == "user":
            query_parts.append(f"[User]: {content}")
        elif role == "assistant":
            tc = msg.get("tool_calls")
            if tc and isinstance(tc, list):
                tc_texts = []
                for call in tc:
                    fn = call.get("function") or {}
                    tc_texts.append(
                        f"调用 {fn.get('name', '?')}({json.dumps(fn.get('arguments', {}), ensure_ascii=False)})"
                    )
                query_parts.append(
                    f"[Assistant]: {content or ''}\n工具调用: {'; '.join(tc_texts)}"
                )
            else:
                query_parts.append(f"[Assistant]: {content}")
        elif role == "tool":
            tool_name = msg.get("name", "tool")
            tool_call_id = msg.get("tool_call_id", "")
            id_info = f" (id={tool_call_id})" if tool_call_id else ""
            query_parts.append(f"[Tool Result - {tool_name}{id_info}]: {content}")

    tool_token: Optional[str] = None
    if tools and tool_choice != "none":
        tool_token = token
        tag_open = build_open_tag(token)
        tag_close = TOOL_CLOSE_TAG

        front_reminder = _build_front_reminder(TOOL_CALL_STRICTNESS, tag_open, tag_close)
        if front_reminder:
            query_parts.insert(0, front_reminder)

        tool_prompt = generate_tool_prompt(
            tools,
            token,
            session.get("prev_tokens"),
            dify_key=dify_key,
        )
        query_parts.append(tool_prompt)
        if tool_choice == "required":
            query_parts.append("\n[重要]: 你必须调用至少一个工具。")
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            fname = (tool_choice.get("function") or {}).get("name")
            if fname:
                query_parts.append(f"\n[重要]: 请使用 {fname} 工具。")

        last_non_system = None
        for m in reversed(selected):
            if (m or {}).get("role") != "system":
                last_non_system = m
                break
        if last_non_system and last_non_system.get("role") == "tool":
            followup = _build_tool_followup_reminder(
                TOOL_CALL_STRICTNESS, tag_open, tag_close
            )
            if followup:
                query_parts.append(followup)

    return "\n\n".join(query_parts), tool_token
