"""
Prompt 方言分派。每个 dialect 模块需暴露:

  render_query(openai_req, session, dify_key) -> (query_text | None, tool_token | None)
  extract_tool_calls(text, token=None, prev_tokens=None, tools=None) -> (clean_text, tool_calls | None)
  OPEN_TAG_PATTERN: re.Pattern   # 用于流式检测工具调用块起始
  HOLDBACK: int                  # 流式输出的安全缓冲字符数 (覆盖最长 open tag)
"""

from importlib import import_module
from typing import Any


def get_dialect(name: str) -> Any:
    if name == "claude":
        return import_module(".claude", __name__)
    if name == "openai":
        return import_module(".openai", __name__)
    return import_module(".generic", __name__)
