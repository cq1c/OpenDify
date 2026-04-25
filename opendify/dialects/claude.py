"""
Claude 方言: 对齐 Anthropic 当前线上提示词的官方风格。

提示词布局 (与 Claude 官方系统提示同形):
    In this environment you have access to a set of tools ...
    You can invoke functions by writing a "<function_calls>" block ...

    <function_calls>
    <invoke name="$FUNCTION_NAME">
    <parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</parameter>
    ...
    </invoke>
    </function_calls>

    String and scalar parameters should be specified as is, while lists
    and objects should use JSON format.

    Here are the functions available in JSONSchema format:
    <functions>
    <function>{"description": "...", "name": "...", "parameters": {...}}</function>
    ...
    </functions>

会话拼接仍用 <system>/<user>/<assistant> 作为 turn marker (Dify
单 query 通道下的本地约定), 工具结果继续用 <function_results><result>。

解析侧兼容 antml: 命名空间前缀以及常见破损/同义写法。
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
# 可选命名空间前缀: 兼容 Claude 内部使用的 antml: 形式
_NS = r"(?:antml:)?"
# 宽松匹配 token 属性: 可选 name=、单/双/无引号、token 也可写成 tok 或省略
_OPEN_ATTR = r'(?:\s+(?:token|tok)\s*=\s*["\']?([^"\'\s>]*)["\']?)?'
# 兼容 function_calls / function_call (单复数) 以及 tool_calls / tool_call 同义写法
_FC_NAMES = rf"{_NS}(?:function_calls?|tool_calls?|fnc|function-calls?)"
_INVOKE_NAME = rf"{_NS}invoke"
_PARAM_NAME = rf"{_NS}(?:parameter|param|arg)"
OPEN_TAG_PATTERN = re.compile(
    rf"<\s*{_FC_NAMES}{_OPEN_ATTR}[^>]*>", re.IGNORECASE
)
_CLOSE_TAG_PATTERN = re.compile(
    rf"<\s*/\s*{_FC_NAMES}\s*>", re.IGNORECASE
)
_BLOCK_PATTERN = re.compile(
    rf"<\s*{_FC_NAMES}{_OPEN_ATTR}[^>]*>(.*?)<\s*/\s*{_FC_NAMES}\s*>",
    re.DOTALL | re.IGNORECASE,
)
# invoke: 接受 <invoke name="x"> / <invoke name='x'> / <invoke name=x> / <invoke x> / <invoke ...>
_INVOKE_PATTERN = re.compile(
    rf'<\s*{_INVOKE_NAME}\s+(?:name\s*=\s*)?["\']?([A-Za-z_][\w\-.]*)["\']?[^>]*>'
    rf'(.*?)<\s*/\s*{_INVOKE_NAME}\s*>',
    re.DOTALL | re.IGNORECASE,
)
# 同上, 但允许丢失 </invoke> 收尾 (后续被另一个 <invoke> 或 </function_calls> 终结)
_INVOKE_LOOSE_PATTERN = re.compile(
    rf'<\s*{_INVOKE_NAME}\s+(?:name\s*=\s*)?["\']?([A-Za-z_][\w\-.]*)["\']?[^>]*>'
    rf'(.*?)(?=<\s*{_INVOKE_NAME}\s|<\s*/\s*{_FC_NAMES}\s*>|\Z)',
    re.DOTALL | re.IGNORECASE,
)
# parameter: 接受 <parameter name="x"> / <parameter name=x> / <parameter x> / <parameter ...>
_PARAM_PATTERN = re.compile(
    rf'<\s*{_PARAM_NAME}\s+(?:name\s*=\s*)?["\']?([A-Za-z_][\w\-.]*)["\']?[^>]*>'
    rf'(.*?)<\s*/\s*{_PARAM_NAME}\s*>',
    re.DOTALL | re.IGNORECASE,
)
# 兜底: 缺失 </parameter> 时, 用下一个 <parameter> 或 </invoke> 截断
_PARAM_LOOSE_PATTERN = re.compile(
    rf'<\s*{_PARAM_NAME}\s+(?:name\s*=\s*)?["\']?([A-Za-z_][\w\-.]*)["\']?[^>]*>'
    rf'(.*?)(?=<\s*{_PARAM_NAME}\s|<\s*/\s*{_INVOKE_NAME}\s*>|<\s*/\s*{_FC_NAMES}\s*>|\Z)',
    re.DOTALL | re.IGNORECASE,
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
    """
    Anthropic 官方风格: 一行 `<function>{JSON}</function>`,
    JSON 体含 description / name / parameters (JSONSchema)。
    """
    name = func.get("name", "unknown")
    raw_desc = func.get("description", "") or ""
    cached = tool_desc_digest.load(name, raw_desc) if raw_desc else None
    if cached:
        desc = cached
    else:
        desc = truncate(raw_desc, TOOL_DESC_MAX_LENGTH * 2)
        if dify_key and raw_desc:
            tool_desc_digest.schedule_generate(dify_key, name, raw_desc)
    obj: Dict[str, Any] = {
        "description": desc,
        "name": name,
        "parameters": func.get("parameters") or {},
    }
    if SIMPLIFIED_TOOL_DEFS:
        body = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    else:
        body = json.dumps(obj, ensure_ascii=False, indent=2)
    return f"<function>{body}</function>"


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

    functions_xml = "<functions>\n" + "\n".join(tool_blocks) + "\n</functions>"

    tag_open = _build_open_tag(token)

    # Anthropic 官方示例: 用 $FUNCTION_NAME / $PARAMETER_NAME / $PARAMETER_VALUE 占位
    example = (
        f"{tag_open}\n"
        '<invoke name="$FUNCTION_NAME">\n'
        '<parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</parameter>\n'
        "...\n"
        "</invoke>\n"
        "...\n"
        "</function_calls>"
    )

    # 头部: 严格对齐 Anthropic 线上 system prompt 的开场白
    intro_lines = [
        "In this environment you have access to a set of tools you can use to "
        "answer the user's question.",
        f'You can invoke functions by writing a "{tag_open}" block like the '
        "following as part of your reply to the user:",
        "",
        example,
        "",
        "String and scalar parameters should be specified as is, while lists "
        "and objects should use JSON format.",
    ]

    # 可选: 令牌防回放 (OpenDify 的扩展, 不属 Anthropic 标准)
    if token and USE_TOOL_TOKEN:
        intro_lines.append("")
        intro_lines.append(
            f'The token attribute on `<function_calls>` for THIS turn is '
            f'"{token}". Do not copy historical or invented tokens.'
        )
        if prev_tokens:
            expired = ", ".join(f'"{t}"' for t in prev_tokens[-5:])
            intro_lines.append(f"(Historical tokens {expired} are now expired.)")

    # 工具清单 (官方风格固定语)
    intro_lines.append("")
    intro_lines.append("Here are the functions available in JSONSchema format:")
    intro_lines.append(functions_xml)

    # 强度分级补充约束 (level >= 2 起加强)
    if level >= 2:
        intro_lines.append("")
        intro_lines.append(
            "Format is mandatory: tool invocations missing the "
            f"`{tag_open}` ... `</function_calls>` envelope will be rejected."
        )
    if level >= 3:
        intro_lines.append(
            "If your response has no `</function_calls>` closing tag, "
            "the entire output is discarded and the user sees nothing."
        )

    return "\n".join(intro_lines).strip()


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


def _extract_params(body: str) -> Dict[str, Any]:
    """先按严格 <parameter>...</parameter> 提取, 没有就用宽松模式兜底。"""
    args: Dict[str, Any] = {}
    matches = list(_PARAM_PATTERN.finditer(body))
    if not matches:
        matches = list(_PARAM_LOOSE_PATTERN.finditer(body))
    for pm in matches:
        pname = pm.group(1).strip()
        if not pname or pname.lower() == "name":
            continue
        pval = _parse_parameter_value(pm.group(2))
        args[pname] = pval
    return args


def _invokes_to_calls(
    invokes_text: str, tools: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    name_map = _tools_by_name(tools)
    calls: List[Dict[str, Any]] = []
    matches = list(_INVOKE_PATTERN.finditer(invokes_text))
    if not matches:
        # 缺 </invoke> 兜底
        matches = list(_INVOKE_LOOSE_PATTERN.finditer(invokes_text))
    for m in matches:
        name = m.group(1).strip()
        if not name:
            continue
        body = m.group(2)
        args = _extract_params(body)
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
    first = _INVOKE_PATTERN.search(text) or _INVOKE_LOOSE_PATTERN.search(text)
    if first:
        clean_text = text[: first.start()].rstrip()
        calls = _invokes_to_calls(text, tools)
        parsed_list = _normalize_tool_calls(calls) if calls else []
        if parsed_list:
            return clean_text, parsed_list

    # ── 3b) 仅有 </function_calls> 收尾, 缺开始标签 ──
    close_match = _CLOSE_TAG_PATTERN.search(text)
    if close_match:
        body = text[: close_match.start()]
        if _INVOKE_PATTERN.search(body) or _INVOKE_LOOSE_PATTERN.search(body):
            inv = _INVOKE_PATTERN.search(body) or _INVOKE_LOOSE_PATTERN.search(body)
            calls = _invokes_to_calls(body, tools)
            parsed_list = _normalize_tool_calls(calls) if calls else []
            if parsed_list:
                clean_text = text[: inv.start()].rstrip()
                return clean_text, parsed_list

    # ── 4) 跨方言兜底: 模型输出了 generic / openai 格式 ──
    from ..tool_calls import extract_tool_calls as _generic_extract
    clean, calls = _generic_extract(text, token, prev_tokens, tools)
    if calls:
        logger.debug("claude dialect: 跨方言兜底 (generic parser) 命中")
        return clean, calls

    return text, None
