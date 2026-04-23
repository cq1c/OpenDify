"""OpenAI ChatCompletion 请求体 → Dify chat-messages 请求体。按 PROMPT_DIALECT 分派。"""

from typing import Any, Dict, Optional

from .config import CONVERSATION_MODE, PROMPT_DIALECT
from .dialects import get_dialect
from .utils import extract_text  # re-export, server.py 仍从此导入

__all__ = ["extract_text", "transform_openai_to_dify"]


def transform_openai_to_dify(
    openai_req: Dict[str, Any],
    session: Dict[str, Any],
    dify_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    dialect = get_dialect(PROMPT_DIALECT)
    query, tool_token = dialect.render_query(
        openai_req, session, dify_key=dify_key
    )
    if not query:
        return None

    dify_req: Dict[str, Any] = {
        "inputs": {},
        "query": query,
        "response_mode": "streaming" if openai_req.get("stream") else "blocking",
        "user": openai_req.get("user", "roo_user"),
    }
    conversation_id = session.get("conversation_id")
    if conversation_id and CONVERSATION_MODE == "auto":
        dify_req["conversation_id"] = conversation_id
    dify_req["_tool_token"] = tool_token
    return dify_req
