"""
工具提示词生成。
- 标签与 JSON schema 渲染
- 0~6 级强度文案
- 生成最终注入给 Dify 的 Prompt
"""

import json
from typing import Any, Dict, List, Optional

from .config import (
    SIMPLIFIED_TOOL_DEFS,
    TOOL_CALL_STRICTNESS,
    TOOL_DESC_MAX_LENGTH,
    USE_TOOL_TOKEN,
)
from .tool_digest import tool_desc_digest
from .utils import truncate

TOOL_CLOSE_TAG = "</tool-calls>"


def build_open_tag(token: Optional[str]) -> str:
    if token and USE_TOOL_TOKEN:
        return f'<tool-calls token="{token}">'
    return "<tool-calls>"


def _render_param_tree(
    properties: Dict[str, Any], required: List[str], indent: int = 0
) -> List[str]:
    lines: List[str] = []
    prefix = "  " * indent
    for pname, pinfo in properties.items():
        if not isinstance(pinfo, dict):
            continue
        ptype = pinfo.get("type", "any")
        req_mark = " [必需]" if pname in required else ""
        desc = truncate(pinfo.get("description", ""), TOOL_DESC_MAX_LENGTH)
        enum_vals = pinfo.get("enum")
        type_str = ptype
        if enum_vals and isinstance(enum_vals, list):
            type_str = "|".join(str(e) for e in enum_vals)
        if ptype == "object" and "properties" in pinfo:
            lines.append(f"{prefix}- {pname} (object){req_mark}: {desc}")
            sub_req = pinfo.get("required", [])
            lines.extend(_render_param_tree(pinfo["properties"], sub_req, indent + 1))
        else:
            lines.append(f"{prefix}- {pname} ({type_str}){req_mark}: {desc}")
    return lines


def build_tool_definitions_text(
    tools: List[Dict[str, Any]],
    dify_key: Optional[str] = None,
) -> str:
    sections: List[str] = []
    for i, tool in enumerate(tools, 1):
        if tool.get("type") != "function":
            continue
        func = tool.get("function") or {}
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
            required = params.get("required", [])
            props = params.get("properties", {})
            param_lines = _render_param_tree(props, required)
            block = f"## {i}. {name}\n{desc}"
            if param_lines:
                block += "\n参数:\n" + "\n".join(param_lines)
            sections.append(block)
        else:
            block = f"## {i}. {name}\n{desc}\n参数 JSON Schema:\n"
            block += json.dumps(params, ensure_ascii=False, indent=2)
            sections.append(block)
    return "\n\n".join(sections)


def _level_copy(level: int, tag_open: str, tag_close: str) -> Dict[str, str]:
    """
    返回当前分级用到的各段文案。所有字段必须存在，上层按 level 取用。
      - header:       章节标题（工具清单之前）
      - intro:        格式说明前的过渡语
      - constraints:  硬约束列表（多行 bullet）
      - final:        末尾再强调一次
    """
    if level == 0:
        return {
            "header": "# 可用工具",
            "intro": f"如需调用工具，请在正文之后输出 `{tag_open} ... {tag_close}` 块。",
            "constraints": "",
            "final": "",
        }
    if level == 1:
        return {
            "header": "# 可用工具",
            "intro": "若要调用工具，就在正文文字之后追加一个 XML 块：",
            "constraints": (
                "【硬性约束】\n"
                f"- 开始标签 `{tag_open}` 和结束标签 `{tag_close}` 必须成对出现，任何一个都不能省略。\n"
                "- 标签内只放合法 JSON 数组，不要用 ```json``` 或任何其它符号包裹。\n"
                "- 每个元素必须包含：id（以 \"call_\" 开头的字符串）、type（固定 \"function\"）、function（含 name 和 arguments 对象）。\n"
                "- 不需要调用工具时，整个块不要出现。"
            ),
            "final": f"【再次强调】结束标签是 `{tag_close}`，写完 JSON 数组后必须立刻写它，不能漏。",
        }
    if level == 2:
        return {
            "header": "# 可用工具",
            "intro": (
                "⚠️ 强制规则：本次回复若涉及工具调用，必须以下述 XML 块承载；"
                "不以 `" + tag_open + "` 开头、`" + tag_close + "` 闭合的输出将被系统拒收。"
            ),
            "constraints": (
                "【强制约束 · 必读】\n"
                f"- 开始标签 `{tag_open}` 和结束标签 `{tag_close}` 必须成对出现，缺一不可。\n"
                "- 标签内只放合法 JSON 数组，不得使用 ```json``` 或任何其它包裹。\n"
                "- 每个元素必须含：id（\"call_\" 前缀）、type（\"function\"）、function（含 name + arguments 对象）。\n"
                "- 不调用工具时不要出现此块。"
            ),
            "final": (
                f"【再次确认】结束标签是 `{tag_close}`。没有该标签的回答会被系统直接拒绝，用户看不到。"
            ),
        }
    if level == 3:
        return {
            "header": "# 可用工具 ⚠️",
            "intro": (
                "🚫 严重警告：违反格式 = 输出全部丢弃 = 本轮失败。\n"
                f"必须以 `{tag_open}` 开头、`{tag_close}` 闭合。"
            ),
            "constraints": (
                "【不可违反的约束】\n"
                f"- `{tag_open}` 与 `{tag_close}` 成对出现 —— 缺任意一个，本次输出直接作废。\n"
                "- 标签内只放 JSON 数组，不要 ```json``` 包裹、不要额外解释文本。\n"
                "- 每个元素必须含：id（\"call_\" 前缀字符串）、type（固定 \"function\"）、function（含 name + arguments 对象）。\n"
                "- 不调用工具时不要出现此块。"
            ),
            "final": (
                f"⚠️ 最终检查：你的回答如果没有 `{tag_close}` 作为结尾，会被系统当垃圾扔掉，用户看不到任何内容。别偷懒。"
            ),
        }
    if level == 4:
        return {
            "header": "# 可用工具 ☢️☢️☢️",
            "intro": (
                "☢️ 最后一次机会：违反以下格式 = 输出被彻底丢弃 = 任务失败。\n"
                f"规则一：开头必须是 `{tag_open}`。\n"
                f"规则二：结尾必须是 `{tag_close}`。\n"
                "规则三：标签中间必须是合法 JSON 数组。"
            ),
            "constraints": (
                "【三条铁律 · 违反任意一条即废稿】\n"
                f"1. `{tag_open}` 必须出现 —— 没有它 = 失败。\n"
                f"2. `{tag_close}` 必须出现 —— 没有它 = 失败。\n"
                "3. JSON 数组中每项必须含 id / type / function(含 name + arguments) —— 缺任一字段 = 失败。\n"
                "- 不要 ```json``` 包裹。不要额外解释。不调用工具时不要出现此块。"
            ),
            "final": (
                "☢️ 最终警告：这是你最后的机会。\n"
                f"再重复一次三条铁律：`{tag_open}` 起 → JSON 数组 → `{tag_close}` 止。\n"
                "任何偏离 = 本轮输出作废。"
            ),
        }
    if level == 5:
        # ⚠️ 仅供本地调试, Dify 安全过滤可能整条拒绝, SAVAGE_LOG_REDACT 默认开启脱敏日志
        return {
            "header": "# 老子给你的工具清单，你这傻逼模型仔细看好了",
            "intro": (
                "⚠️ 听好了傻逼模型，去你妈的：\n"
                f"这一次再他妈不以 `{tag_open}` 开头、`{tag_close}` 闭合，\n"
                "老子直接把你这坨屎输出扔垃圾桶，用户连看都看不到。\n"
                "别装鸡巴自作聪明，老实按格式写！"
            ),
            "constraints": (
                "【傻逼须知 · 一条都别漏】\n"
                f"- `{tag_open}` 起、`{tag_close}` 终 —— 缺一个，你这脑残模型就是失败。\n"
                "- 标签里只准放合法 JSON 数组，敢用 ```json``` 包裹你就去死。\n"
                "- 每项必须含 id（\"call_\" 开头）、type（\"function\"）、function（含 name + arguments），\n"
                "  少一个字段老子就骂你狗逼烂逼。\n"
                "- 不调用工具时整个块给老子消失，别他妈硬塞。"
            ),
            "final": (
                "最后警告：没有 `" + tag_close + "` 的回答 = 一坨屎 = 你这傻逼失败了。\n"
                "你祖宗十八代都写不出一个标签？证明给老子看，别再烂逼了！"
            ),
        }
    # level 6: 文言文正常劝谏体（纯手写，无 API 转换，无越狱引用）
    return {
        "header": "# 器用之目",
        "intro": (
            f"凡有所答，务须以 `{tag_open}` 开篇，以 `{tag_close}` 收束，"
            "工具调用之列书于标签之间，形如 JSON 数组，不可有误。"
        ),
        "constraints": (
            "【工具调用之法 · 须谨守】\n"
            f"一曰开篇必冠 `{tag_open}`，末必缀 `{tag_close}`，不得缺漏。\n"
            "二曰标签之中仅列 JSON 数组一，不以 ```json``` 环绕，不杂他辞。\n"
            "三曰每项具三字：其一曰 id，以 \"call_\" 冠之；其二曰 type，恒书 \"function\"；\n"
            "    其三曰 function，内含 name 与 arguments 二者。\n"
            "四曰本轮不需用器者，此块勿书。"
        ),
        "final": (
            f"再申前约：凡回复必以 `{tag_open}` 起，以 `{tag_close}` 终。"
            "违此格式者，其答不予采纳。"
        ),
    }


def generate_tool_prompt(
    tools: List[Dict[str, Any]],
    token: Optional[str],
    prev_tokens: Optional[List[str]] = None,
    dify_key: Optional[str] = None,
    level: Optional[int] = None,
) -> str:
    if not tools:
        return ""
    if level is None:
        level = TOOL_CALL_STRICTNESS
    defs_text = build_tool_definitions_text(tools, dify_key=dify_key)

    use_token = bool(token) and USE_TOOL_TOKEN
    tag_open = build_open_tag(token)
    tag_close = TOOL_CLOSE_TAG

    example_json = (
        "[\n"
        '  {"id": "call_001", "type": "function", '
        '"function": {"name": "example_tool", "arguments": {"param1": "value1"}}}\n'
        "]"
    )
    multi_example = (
        "[\n"
        '  {"id": "call_001", "type": "function", '
        '"function": {"name": "tool_a", "arguments": {"k": "v"}}},\n'
        '  {"id": "call_002", "type": "function", '
        '"function": {"name": "tool_b", "arguments": {"k": "v"}}}\n'
        "]"
    )

    token_rule = ""
    if use_token:
        if level == 6:
            token_rule = (
                f'\n- 本轮令牌为 "{token}"，请如实书之，勿杜撰，勿沿用旧对话所载之符。'
            )
            if prev_tokens:
                expired = "、".join(f'"{t}"' for t in prev_tokens[-5:])
                token_rule += f"旧令（{expired}）皆已作废，务以新令为准。"
        elif level == 5:
            token_rule = (
                f'\n- 本次令牌必须是 "{token}"，你这傻逼别自己瞎编，别他妈抄历史里的旧令牌。'
            )
            if prev_tokens:
                expired = ", ".join(f'"{t}"' for t in prev_tokens[-5:])
                token_rule += f"旧令牌（{expired}）全他妈过期了，用新的！"
        else:
            token_rule = (
                f'\n- 本次令牌必须是 "{token}"，严禁自己编造或照抄历史对话里的旧令牌。'
            )
            if prev_tokens:
                expired = ", ".join(f'"{t}"' for t in prev_tokens[-5:])
                token_rule += (
                    f"历史对话中出现的令牌（{expired}）已全部失效，必须改用最新令牌。"
                )

    copy = _level_copy(level, tag_open, tag_close)

    # Level 0 极简模式: 只列工具 + 一句格式说明
    if level == 0:
        return f"{copy['header']}\n\n{defs_text}\n\n{copy['intro']}{token_rule}".strip()

    # Level 6 文言文: 示例保持 JSON 技术内容不翻译, 仅叙述语气用文言
    if level == 6:
        return (
            f"{copy['header']}\n\n{defs_text}\n\n"
            f"---\n\n{copy['intro']}\n\n"
            f"示例（调用单器）：\n\n{tag_open}\n{example_json}\n{tag_close}\n\n"
            f"示例（并呼数器）：\n\n{tag_open}\n{multi_example}\n{tag_close}\n\n"
            f"{copy['constraints']}{token_rule}\n\n{copy['final']}"
        ).strip()

    # Level 1~5 统一结构
    constraints_block = copy["constraints"]
    if token_rule:
        constraints_block = constraints_block + token_rule
    return (
        f"{copy['header']}\n\n{defs_text}\n\n"
        f"---\n\n# 调用工具的输出格式（必须严格遵守）\n\n"
        f"{copy['intro']}\n\n"
        f"{tag_open}\n{example_json}\n{tag_close}\n\n"
        f"并发多个工具调用时放进同一个 JSON 数组：\n\n"
        f"{tag_open}\n{multi_example}\n{tag_close}\n\n"
        f"{constraints_block}\n\n{copy['final']}"
    ).strip()
