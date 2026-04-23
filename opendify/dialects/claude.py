"""
Claude 方言: Anthropic 原生 XML 风格。

Query 文本:
  <system>...</system>
  <tools><tool>...</tool></tools>
  <user>...</user>
  <assistant>... <function_calls><invoke name="X"><parameter name="p">v</parameter></invoke></function_calls></assistant>
  <user><function_results><result>...</result></function_results></user>

期望模型输出:
  <function_calls [token="XXX"]>
  <invoke name="tool_x">
  <parameter name="p">value</parameter>
  </invoke>
  </function_calls>
"""

import json
import re
import secrets
from typing import Any, Dict, List, Optional, Tuple

from ..config import (
    CONVERSATION_MODE,
    ONLY_RECENT_MESSAGES,
    STRIP_SYSTEM_AFTER_FIRST,
    SIMPLIFIED_TOOL_DEFS,
    SYSTEM_PROMPT_MAX_LENGTH,
    TOOL_CALL_STRICTNESS,
    TOOL_DESC_MAX_LENGTH,
    USE_TOOL_TOKEN,
    logger,
)
from ..tool_calls import (
    _coerce_arguments,
    _normalize_tool_calls,
    _robust_json_parse,
    _tools_by_name,
)
from ..tool_digest import tool_desc_digest
from ..utils import extract_text, truncate

CLOSE_TAG = "</function_calls>"
_OPEN_ATTR = r'(?:\s+token="([^"]*)")?'
OPEN_TAG_PATTERN = re.compile(rf"<function_calls{_OPEN_ATTR}\s*>", re.IGNORECASE)
_BLOCK_PATTERN = re.compile(
    rf"<function_calls{_OPEN_ATTR}\s*>(.*?)</function_calls>",
    re.DOTALL | re.IGNORECASE,
)
_INVOKE_PATTERN = re.compile(
    r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>', re.DOTALL | re.IGNORECASE
)
_PARAM_PATTERN = re.compile(
    r'<parameter\s+name="([^"]+)"\s*>(.*?)</parameter>', re.DOTALL | re.IGNORECASE
)
HOLDBACK = 55  # `<function_calls token="XXXXXX">` 约 35 字, 预留余量


# ────────────────────────────────────────
#  渲染
# ────────────────────────────────────────


def _build_open_tag(token: Optional[str]) -> str:
    if token and USE_TOOL_TOKEN:
        return f'<function_calls token="{token}">'
    return "<function_calls>"


def _render_tool(
    func: Dict[str, Any], dify_key: Optional[str] = None
) -> str:
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
    if SIMPLIFIED_TOOL_DEFS:
        schema_str = json.dumps(params, ensure_ascii=False, separators=(",", ":"))
    else:
        schema_str = json.dumps(params, ensure_ascii=False, indent=2)
    return (
        f"<tool>\n"
        f"<name>{name}</name>\n"
        f"<description>{desc}</description>\n"
        f"<input_schema>\n{schema_str}\n</input_schema>\n"
        f"</tool>"
    )


def _build_tool_prompt(
    tools: List[Dict[str, Any]],
    token: Optional[str],
    prev_tokens: Optional[List[str]],
    dify_key: Optional[str],
    level: int,
) -> str:
    tool_blocks: List[str] = []
    for t in tools:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        tool_blocks.append(_render_tool(fn, dify_key=dify_key))

    tools_xml = "<tools>\n" + "\n".join(tool_blocks) + "\n</tools>"

    tag_open = _build_open_tag(token)

    example = (
        f"{tag_open}\n"
        '<invoke name="example_tool">\n'
        '<parameter name="p1">value1</parameter>\n'
        "</invoke>\n"
        "</function_calls>"
    )

    if level <= 0:
        intro = (
            f"若需调用工具, 在正文之后输出 `{tag_open} ... </function_calls>` 块。"
            "每个 `<parameter>` 内为该参数的值 (字符串原样, 复杂类型写 JSON)。"
        )
        token_note = ""
        if token and USE_TOOL_TOKEN:
            token_note = f' 本次令牌为 "{token}"。'
        return f"{tools_xml}\n\n{intro}{token_note}".strip()

    constraints = [
        f"- 工具调用必须包裹在 `{tag_open}` 与 `</function_calls>` 之间, 缺一不可。",
        '- 每个调用写一个 `<invoke name="工具名">...</invoke>`, 可并列多个 invoke。',
        '- 参数用 `<parameter name="参数名">值</parameter>`, 字符串原样写, 数字/布尔/数组/对象请写合法 JSON 字面量。',
        "- 无需调用工具时整个 `<function_calls>` 块不要出现。",
    ]
    if token and USE_TOOL_TOKEN:
        constraints.append(
            f'- 本次令牌必须是 "{token}", 严禁照抄历史或自行编造。'
        )
        if prev_tokens:
            expired = ", ".join(f'"{t}"' for t in prev_tokens[-5:])
            constraints.append(f"  (历史令牌 {expired} 已全部作废)")

    header = "# 可用工具"
    intro = "若需调用工具, 在正文之后追加一个 XML 块:"
    final = ""
    if level >= 2:
        header = "# 可用工具 ⚠️"
        intro = (
            f"⚠️ 强制格式: 涉及工具调用时, 必须以 `{tag_open}` 开头、"
            f"`</function_calls>` 结尾, 违反将被拒收。"
        )
    if level >= 3:
        final = (
            f"\n\n⚠️ 最终检查: 没有 `</function_calls>` 收尾的响应会被系统丢弃, "
            "用户看不到任何内容。"
        )

    return (
        f"{header}\n\n{tools_xml}\n\n"
        f"---\n\n{intro}\n\n{example}\n\n"
        "约束:\n" + "\n".join(constraints) + final
    ).strip()


def _render_assistant(msg: Dict[str, Any]) -> str:
    content = extract_text(msg.get("content"))
    tc = msg.get("tool_calls")
    if tc and isinstance(tc, list):
        invokes: List[str] = []
        for call in tc:
            fn = call.get("function") or {}
            name = fn.get("name", "unknown")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            if not isinstance(args, dict):
                args = {}
            param_lines: List[str] = []
            for k, v in args.items():
                if isinstance(v, str):
                    param_lines.append(f'<parameter name="{k}">{v}</parameter>')
                else:
                    param_lines.append(
                        f'<parameter name="{k}">{json.dumps(v, ensure_ascii=False)}</parameter>'
                    )
            params_xml = "\n".join(param_lines)
            invokes.append(
                f'<invoke name="{name}">\n{params_xml}\n</invoke>'
                if params_xml
                else f'<invoke name="{name}">\n</invoke>'
            )
        body = content + ("\n" if content else "")
        body += "<function_calls>\n" + "\n".join(invokes) + "\n</function_calls>"
        return f"<assistant>\n{body}\n</assistant>"
    return f"<assistant>\n{content}\n</assistant>"


def _render_tool_result(msg: Dict[str, Any]) -> str:
    content = extract_text(msg.get("content"))
    name = msg.get("name", "tool")
    call_id = msg.get("tool_call_id", "")
    id_xml = f"<tool_use_id>{call_id}</tool_use_id>\n" if call_id else ""
    return (
        "<user>\n<function_results>\n<result>\n"
        f"{id_xml}"
        f"<name>{name}</name>\n"
        f"<stdout>\n{content}\n</stdout>\n"
        "</result>\n</function_results>\n</user>"
    )


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

    # system 内容合并进单个 <system> 块
    system_texts: List[str] = []
    convo_parts: List[str] = []
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
            convo_parts.append(f"<user>\n{extract_text(msg.get('content'))}\n</user>")
        elif role == "assistant":
            convo_parts.append(_render_assistant(msg))
        elif role == "tool":
            convo_parts.append(_render_tool_result(msg))

    sections: List[str] = []
    if system_texts:
        sections.append("<system>\n" + "\n\n".join(system_texts) + "\n</system>")

    tool_token: Optional[str] = None
    if tools and tool_choice != "none":
        tool_token = token
        tool_prompt = _build_tool_prompt(
            tools, token, session.get("prev_tokens"), dify_key, TOOL_CALL_STRICTNESS
        )
        sections.append(tool_prompt)
        if tool_choice == "required":
            sections.append("[重要]: 本轮你必须调用至少一个工具。")
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            fname = (tool_choice.get("function") or {}).get("name")
            if fname:
                sections.append(f"[重要]: 请使用 {fname} 工具。")

    sections.extend(convo_parts)

    if tool_token:
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
            sections.append(
                "[注意]: 上一步刚用过工具, 这一步多半需要继续调用。"
                "请以 `<function_calls>` 块承载。"
            )

    return "\n\n".join(sections), tool_token


# ────────────────────────────────────────
#  解析
# ────────────────────────────────────────


def _parse_parameter_value(raw: str) -> Any:
    """
    参数值优先按 JSON 字面量解析 (数字/布尔/数组/对象), 失败则当字符串。
    """
    s = raw.strip()
    if not s:
        return ""
    # 明确是字符串字面量 → 保持字符串 (去掉外层引号由 JSON 解析完成)
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("{") or s.startswith("[")
    ) or s in ("true", "false", "null"):
        parsed = _robust_json_parse(s)
        if parsed is not None or s == "null":
            return parsed
    # 数字
    try:
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        if re.fullmatch(r"-?\d+\.\d+", s):
            return float(s)
    except Exception:
        pass
    return raw  # 字符串原样 (保留前后空白也无妨, schema coerce 会处理)


def _invokes_to_calls(
    invokes_text: str, tools: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    name_map = _tools_by_name(tools)
    calls: List[Dict[str, Any]] = []
    for m in _INVOKE_PATTERN.finditer(invokes_text):
        name = m.group(1).strip()
        body = m.group(2)
        args: Dict[str, Any] = {}
        for pm in _PARAM_PATTERN.finditer(body):
            pname = pm.group(1).strip()
            pval = _parse_parameter_value(pm.group(2))
            args[pname] = pval
        spec = name_map.get(name)
        if isinstance(spec, dict):
            schema = spec.get("parameters") or {}
            if isinstance(schema, dict):
                args = _coerce_arguments(args, schema)
        calls.append(
            {
                "id": f"call_{secrets.token_hex(8)}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        )
    return calls


def extract_tool_calls(
    text: str,
    token: Optional[str] = None,
    prev_tokens: Optional[List[str]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """
    从模型输出中提取工具调用。分级兜底:
      1. 完整 <function_calls>...</function_calls> 块
      2. 只有 <function_calls> 开始标签 → 取后续文本作 invoke 体
      3. 无外壳但有裸 <invoke> → 直接扫描
      4. 跨方言兜底: 尝试 generic <tool-calls>[JSON] 和 <tool_call>{JSON} 格式
    """
    if not text:
        return text or "", None

    # ── 1) 完整块 ──
    match = _BLOCK_PATTERN.search(text)
    if match:
        matched_token = match.group(1)
        if USE_TOOL_TOKEN and token and matched_token and matched_token != token:
            if not prev_tokens or matched_token not in prev_tokens:
                logger.debug(
                    "function_calls token 不匹配 (got=%s expected=%s), 宽松接受",
                    matched_token,
                    token,
                )
        clean_text = text[: match.start()].rstrip()
        calls = _invokes_to_calls(match.group(2), tools)
        parsed_list = _normalize_tool_calls(calls) if calls else []
        return clean_text, (parsed_list or None)

    # ── 2) 只有开始标签 ──
    open_match = OPEN_TAG_PATTERN.search(text)
    if open_match:
        clean_text = text[: open_match.start()].rstrip()
        tail = text[open_match.end() :]
        calls = _invokes_to_calls(tail, tools)
        parsed_list = _normalize_tool_calls(calls) if calls else []
        if parsed_list:
            return clean_text, parsed_list

    # ── 3) 裸 invoke (模型忘写外壳) ──
    if _INVOKE_PATTERN.search(text):
        first = _INVOKE_PATTERN.search(text)
        clean_text = text[: first.start()].rstrip() if first else text
        calls = _invokes_to_calls(text, tools)
        parsed_list = _normalize_tool_calls(calls) if calls else []
        if parsed_list:
            return clean_text, parsed_list

    # ── 4) 跨方言兜底: 模型输出了 generic / openai 格式 ──
    from ..tool_calls import extract_tool_calls as _generic_extract
    clean, calls = _generic_extract(text, token, prev_tokens, tools)
    if calls:
        logger.debug("claude dialect: 跨方言兜底 (generic parser) 命中")
        return clean, calls

    return text, None
