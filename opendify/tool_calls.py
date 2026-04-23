"""
工具调用提取：把模型返回文本里的 `<tool-calls>` XML 块解析为 OpenAI tool_calls 列表。
包含 JSON 宽松解析、参数类型按 schema 修正、激进兜底。
"""

import json
import re
import secrets
from typing import Any, Dict, List, Optional, Tuple

import json_repair

from .config import AGGRESSIVE_TOOL_RECOVERY, USE_TOOL_TOKEN, logger
from .tool_prompt import TOOL_CLOSE_TAG

_TOOL_TAG_PATTERN = re.compile(
    r'<tool-calls(?:\s+token="([^"]*)")?\s*>(.*?)</tool-calls>',
    re.DOTALL | re.IGNORECASE,
)
TOOL_OPEN_TAG_PATTERN = re.compile(
    r'<tool-calls(?:\s+token="([^"]*)")?\s*>',
    re.IGNORECASE,
)


def _robut_json_loads(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json_repair.loads(text)
    except Exception as e:
        logger.warning(f"JSON repair failed: {e}")
    return None


def _robust_json_parse(text: str) -> Optional[Any]:
    text = text.strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    if ret := _robut_json_loads(text):
        return ret
    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    if ret := _robut_json_loads(fixed):
        return ret
    stack: List[str] = []
    for ch in text:
        if ch in "[{":
            stack.append("]" if ch == "[" else "}")
        elif ch in "]}":
            if stack and stack[-1] == ch:
                stack.pop()
    if stack:
        patched = text + "".join(reversed(stack))
        if ret := _robut_json_loads(patched):
            return ret
        patched_fixed = re.sub(r",\s*([}\]])", r"\1", patched)
        if ret := _robut_json_loads(patched_fixed):
            return ret
    return None


def _coerce_value(value: Any, schema: Optional[Dict[str, Any]]) -> Any:
    """
    按 JSON schema 把 value 修复到期望类型。宽松——无法修复时原样返回。
    覆盖弱模型最常见的错配：
      - array<object> 收到 array<string>：用 required[0] 或第一个 property 包装
      - array 收到单值：包成 [value]
      - string/number/integer/boolean 常见跨类型
    """
    if not isinstance(schema, dict):
        return value
    t = schema.get("type")

    if t == "array":
        item_schema = schema.get("items") if isinstance(schema.get("items"), dict) else {}
        if not isinstance(value, list):
            value = [value]
        return [_coerce_value(it, item_schema) for it in value]

    if t == "object":
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        if isinstance(value, dict):
            return {
                k: (_coerce_value(v, props[k]) if k in props else v)
                for k, v in value.items()
            }
        if isinstance(value, (str, int, float, bool)) and props:
            wrap_key: Optional[str] = None
            if required and required[0] in props:
                wrap_key = required[0]
            else:
                wrap_key = next(iter(props.keys()))
            sub = props.get(wrap_key, {})
            return {wrap_key: _coerce_value(value, sub)}
        return value

    if t == "string":
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    if t in ("number", "integer"):
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value) if t == "integer" else value
        if isinstance(value, str):
            s = value.strip()
            try:
                return int(s) if t == "integer" else float(s)
            except ValueError:
                try:
                    # "1.0" -> int(1)
                    return int(float(s)) if t == "integer" else float(s)
                except ValueError:
                    return value
        return value

    if t == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("true", "1", "yes", "y"):
                return True
            if s in ("false", "0", "no", "n", ""):
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return value

    return value


def _coerce_arguments(
    args: Dict[str, Any], parameters_schema: Dict[str, Any]
) -> Dict[str, Any]:
    props = parameters_schema.get("properties") or {}
    if not isinstance(props, dict) or not props:
        return args
    return {
        k: (_coerce_value(v, props[k]) if k in props else v) for k, v in args.items()
    }


def _tools_by_name(
    tools: Optional[List[Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for t in tools or []:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            out[name] = fn
    return out


def _coerce_tool_calls_parsed(
    parsed: Any, tools: Optional[List[Dict[str, Any]]]
) -> Any:
    """
    在 normalize 之前把每个 tool_call 的 arguments 按对应工具 schema 修整。
    处理 dict（含 tool_calls 键）、list 两种 parsed 形态。
    """
    if not tools or parsed is None:
        return parsed

    name_map = _tools_by_name(tools)
    if not name_map:
        return parsed

    def _fix_one(call: Any) -> Any:
        if not isinstance(call, dict):
            return call
        fn = call.get("function")
        if not isinstance(fn, dict):
            return call
        name = fn.get("name")
        spec = name_map.get(name) if name else None
        if not isinstance(spec, dict):
            return call
        schema = spec.get("parameters") or {}
        if not isinstance(schema, dict):
            return call
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return call
        if not isinstance(args, dict):
            return call
        new_args = _coerce_arguments(args, schema)
        new_call = dict(call)
        new_fn = dict(fn)
        new_fn["arguments"] = new_args
        new_call["function"] = new_fn
        return new_call

    if isinstance(parsed, dict):
        if isinstance(parsed.get("tool_calls"), list):
            new = dict(parsed)
            new["tool_calls"] = [_fix_one(c) for c in parsed["tool_calls"]]
            return new
        return _fix_one(parsed)
    if isinstance(parsed, list):
        return [_fix_one(c) for c in parsed]
    return parsed


def _normalize_tool_calls(raw: Any) -> List[Dict[str, Any]]:
    items: List[Any] = []
    if isinstance(raw, dict):
        tc = raw.get("tool_calls")
        items = tc if isinstance(tc, list) else [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        return []
    result: List[Dict[str, Any]] = []
    for tc in items:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") or {}
        if not isinstance(func, dict):
            continue
        name = func.get("name")
        if not name:
            continue
        args = func.get("arguments")
        if args is None:
            args = {}
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False, indent=2)
        call_id = tc.get("id") or f"call_{secrets.token_hex(8)}"
        result.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        )
    return result


def _looks_like_tool_calls(parsed: Any) -> bool:
    """判定一个解析结果是否像 tool_calls 结构。"""
    if not isinstance(parsed, list) or not parsed:
        return False
    first = parsed[0]
    if not isinstance(first, dict):
        return False
    fn = first.get("function")
    return isinstance(fn, dict) and bool(fn.get("name"))


def _slice_balanced_array(text: str, end_inclusive: int) -> Optional[Tuple[int, int]]:
    """
    从 text[end_inclusive] 这个 ']' 向前找到匹配的 '['（括号平衡），
    返回 (start, end+1) 切片下标。失败返回 None。
    """
    if end_inclusive < 0 or end_inclusive >= len(text) or text[end_inclusive] != "]":
        return None
    depth = 0
    for i in range(end_inclusive, -1, -1):
        ch = text[i]
        if ch == "]":
            depth += 1
        elif ch == "[":
            depth -= 1
            if depth == 0:
                return (i, end_inclusive + 1)
    return None


def extract_tool_calls(
    text: str,
    token: Optional[str] = None,
    prev_tokens: Optional[List[str]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """
    从模型输出中提取 tool_calls。宽松策略（弱模型友好）：
      1) 首选完整的 <tool-calls [token="..."]>...</tool-calls> 对
      2) 只有开始标签没有结束标签 → 取标签之后到末尾为 JSON
      3) 只有结束标签没有开始标签 → 向前找最近的合法 JSON 数组
      4) 两端都没有但末尾像合法的 tool_calls JSON 数组 → 宽松接受
    若传入 tools，则按工具 parameters schema 修复 arguments 类型错配。
    对 token 只做"记录"，不做拒绝（弱模型常会抄错）。
    """
    if not text:
        return text or "", None

    def _finalize(parsed: Any) -> Optional[List[Dict[str, Any]]]:
        if parsed is None:
            return None
        parsed = _coerce_tool_calls_parsed(parsed, tools)
        calls = _normalize_tool_calls(parsed)
        return calls or None

    # ── 1) 完整标签对 ──
    match = _TOOL_TAG_PATTERN.search(text)
    if match:
        matched_token = match.group(1)
        if USE_TOOL_TOKEN and token and matched_token and matched_token != token:
            if not prev_tokens or matched_token not in prev_tokens:
                logger.debug(
                    "tool-calls token 不匹配（got=%s expected=%s），宽松接受",
                    matched_token,
                    token,
                )
        clean_text = text[: match.start()].rstrip()
        calls = _finalize(_robust_json_parse(match.group(2)))
        if calls:
            return clean_text, calls
        return clean_text, None

    # ── 2) 有开始标签、无结束标签 ──
    open_match = TOOL_OPEN_TAG_PATTERN.search(text)
    if open_match:
        clean_text = text[: open_match.start()].rstrip()
        tail = text[open_match.end() :].strip()
        calls = _finalize(_robust_json_parse(tail))
        if calls:
            return clean_text, calls

    # ── 3) 无开始标签、有结束标签 ──
    close_pos = text.find(TOOL_CLOSE_TAG)
    if close_pos >= 0:
        before = text[:close_pos]
        last_bracket = before.rfind("]")
        span = _slice_balanced_array(before, last_bracket) if last_bracket >= 0 else None
        if span:
            start, end = span
            calls = _finalize(_robust_json_parse(text[start:end]))
            if calls:
                return text[:start].rstrip(), calls

    # ── 4) 两端都无但末尾有合法 tool_calls JSON ──
    last_bracket = text.rfind("]")
    if last_bracket >= 0:
        span = _slice_balanced_array(text, last_bracket)
        if span:
            start, end = span
            parsed = _robust_json_parse(text[start:end])
            if _looks_like_tool_calls(parsed):
                calls = _finalize(parsed)
                if calls:
                    return text[:start].rstrip(), calls

    # ── 5) 激进兜底: 扫描 tool_name({...}) 样式, 对白名单内的工具名做抢救 ──
    if AGGRESSIVE_TOOL_RECOVERY and tools:
        recovered, clean_text = _aggressive_recover(text, tools)
        if recovered:
            calls = _finalize(recovered)
            if calls:
                return clean_text, calls

    return text, None


def _aggressive_recover(
    text: str, tools: List[Dict[str, Any]]
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """
    激进抢救: 当四级兜底全失败时, 扫描形如 tool_name({...}) 的片段。
    仅当 tool_name 命中 tools 白名单才接受, 避免把模型的普通叙述误判成工具调用。
    返回 (recovered_list, clean_text)。失败返回 (None, text)。
    """
    name_map = _tools_by_name(tools)
    if not name_map:
        return None, text

    # 匹配 "tool_name ({ ... })" 或 "tool_name ( { ... } )"
    pattern = re.compile(
        r"(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*(?P<body>\{.*?\})\s*\)",
        re.DOTALL,
    )
    results: List[Dict[str, Any]] = []
    first_start: Optional[int] = None
    for m in pattern.finditer(text):
        name = m.group("name")
        if name not in name_map:
            continue
        body = m.group("body")
        parsed_args = _robust_json_parse(body)
        if not isinstance(parsed_args, dict):
            continue
        if first_start is None:
            first_start = m.start()
        results.append(
            {
                "id": f"call_{secrets.token_hex(4)}",
                "type": "function",
                "function": {"name": name, "arguments": parsed_args},
            }
        )
    if not results:
        return None, text
    clean_text = text[:first_start].rstrip() if first_start is not None else text
    return results, clean_text
