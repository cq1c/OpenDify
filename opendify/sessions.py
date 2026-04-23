"""会话管理：按 (model, system prompt) 维度追踪 Dify conversation_id、工具令牌、累计 usage。"""

import hashlib
import secrets
from typing import Any, Dict


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _make_key(model: str, system_text: str) -> str:
        digest = hashlib.md5((model + system_text[:300]).encode()).hexdigest()[:12]
        return f"{model}::{digest}"

    def get(self, model: str, system_text: str) -> Dict[str, Any]:
        key = self._make_key(model, system_text)
        if key in self._sessions:
            session = self._sessions[key]
            session["msg_count"] += 1
            return session
        token = secrets.token_hex(3)
        session: Dict[str, Any] = {
            "key": key,
            "token": token,
            "conversation_id": None,
            "msg_count": 1,
            "prev_tokens": [],
            "last_non_system_count": 0,
            # 累积 usage：在 conversation 模式下代表整个任务跨多轮的总 token 数。
            "cum_prompt_tokens": 0,
            "cum_completion_tokens": 0,
            "cum_total_tokens": 0,
        }
        self._sessions[key] = session
        return session

    def update_conversation_id(self, session: Dict[str, Any], cid: str) -> None:
        if cid:
            session["conversation_id"] = cid
            self._sessions[session["key"]] = session

    def rotate_token(self, session: Dict[str, Any]) -> str:
        old = session["token"]
        prev = (session.get("prev_tokens") or []) + [old]
        session["prev_tokens"] = prev[-5:]
        new_token = secrets.token_hex(3)
        session["token"] = new_token
        return new_token

    def maybe_reset_for_new_task(
        self, session: Dict[str, Any], non_system_msg_count: int
    ) -> bool:
        """
        启发式检测 Roo Code 新任务：非 system 消息数从 >1 回落到 1 时视为新任务。
        触发时清空 conversation_id、消息计数、历史 token，但不立即生成新 token —
        由外层 rotate_token 统一负责。
        """
        last = session.get("last_non_system_count", 0)
        is_new_task = non_system_msg_count <= 1 and last > 1
        if is_new_task:
            session["conversation_id"] = None
            session["msg_count"] = 1
            session["prev_tokens"] = []
            session["cum_prompt_tokens"] = 0
            session["cum_completion_tokens"] = 0
            session["cum_total_tokens"] = 0
        session["last_non_system_count"] = non_system_msg_count
        return is_new_task

    def accumulate_usage(
        self, session: Dict[str, Any], usage: Dict[str, Any]
    ) -> Dict[str, int]:
        """把本轮 usage 累加到 session 的累计字段并返回累计值。"""
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
        session["cum_prompt_tokens"] = session.get("cum_prompt_tokens", 0) + prompt
        session["cum_completion_tokens"] = (
            session.get("cum_completion_tokens", 0) + completion
        )
        session["cum_total_tokens"] = session.get("cum_total_tokens", 0) + total
        return {
            "prompt_tokens": session["cum_prompt_tokens"],
            "completion_tokens": session["cum_completion_tokens"],
            "total_tokens": session["cum_total_tokens"],
        }


sessions = SessionStore()
