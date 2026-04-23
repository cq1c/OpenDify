"""
请求/响应日志记录器。
将 OpenAI 客户端请求、Dify 转发请求、Dify 响应、OpenAI 返回响应
记录到独立的日志文件中，格式化 JSON 便于阅读调试。
"""

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    REQUEST_LOG_BACKUP_COUNT,
    REQUEST_LOG_DIR,
    REQUEST_LOG_ENABLED,
    REQUEST_LOG_MAX_BODY,
    REQUEST_LOG_MAX_SIZE,
    SAVAGE_LOG_REDACT,
    TOOL_CALL_STRICTNESS,
)


class RequestResponseLogger:
    def __init__(self) -> None:
        self._logger: Optional[logging.Logger] = None
        if REQUEST_LOG_ENABLED:
            self._setup()

    def _setup(self) -> None:
        log_dir = Path(REQUEST_LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger("opendify.traffic")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        # 清除已有 handler 防止重复
        self._logger.handlers.clear()

        handler = RotatingFileHandler(
            str(log_dir / "traffic.log"),
            maxBytes=REQUEST_LOG_MAX_SIZE,
            backupCount=REQUEST_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

    @property
    def enabled(self) -> bool:
        return self._logger is not None

    def _maybe_redact_savage(self, body: Any) -> Any:
        """Level 5 savage 模式下，对日志中出现的脏话做脱敏替换。"""
        if TOOL_CALL_STRICTNESS != 5 or not SAVAGE_LOG_REDACT:
            return body
        sensitive = (
            "妈逼", "傻逼", "鸡巴", "屌", "去死", "脑残",
            "祖宗", "狗逼", "烂逼", "他妈", "老子",
        )
        placeholder = "[SAVAGE_PROMPT_REDACTED]"

        def _scrub(s: str) -> str:
            if any(w in s for w in sensitive):
                return placeholder
            return s

        if isinstance(body, str):
            return _scrub(body)
        if isinstance(body, dict):
            new: Dict[str, Any] = {}
            for k, v in body.items():
                if isinstance(v, str):
                    new[k] = _scrub(v)
                else:
                    new[k] = self._maybe_redact_savage(v)
            return new
        if isinstance(body, list):
            return [self._maybe_redact_savage(v) for v in body]
        return body

    def _truncate_body(self, body: Any) -> Any:
        """可选脱敏 + 截断超大 body"""
        body = self._maybe_redact_savage(body)
        if REQUEST_LOG_MAX_BODY <= 0:
            return body
        if isinstance(body, str) and len(body) > REQUEST_LOG_MAX_BODY:
            return body[:REQUEST_LOG_MAX_BODY] + f"... [截断，总长 {len(body)}]"
        if isinstance(body, dict):
            text = json.dumps(body, ensure_ascii=False)
            if len(text) > REQUEST_LOG_MAX_BODY:
                return json.loads(text[:REQUEST_LOG_MAX_BODY] + "}")  # best-effort
        return body

    def _format_entry(self, entry: Dict[str, Any]) -> str:
        sep = "=" * 72
        return f"\n{sep}\n{json.dumps(entry, ensure_ascii=False, indent=2)}\n{sep}\n"

    def log_openai_request(
        self, request_id: str, model: str, body: Dict[str, Any]
    ) -> None:
        if not self._logger:
            return
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": "CLIENT → PROXY",
            "stage": "openai_request",
            "request_id": request_id,
            "model": model,
            "stream": body.get("stream", False),
            "message_count": len(body.get("messages", [])),
            "tools_count": len(body.get("tools", [])),
            "tool_choice": body.get("tool_choice"),
            "body": self._truncate_body(body),
        }
        self._logger.debug(self._format_entry(entry))

    def log_dify_request(
        self,
        request_id: str,
        endpoint: str,
        body: Dict[str, Any],
        conversation_id: Optional[str],
    ) -> None:
        if not self._logger:
            return
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": "PROXY → DIFY",
            "stage": "dify_request",
            "request_id": request_id,
            "endpoint": endpoint,
            "conversation_id": conversation_id,
            "query_length": len(body.get("query", "")),
            "response_mode": body.get("response_mode"),
            "body": self._truncate_body(body),
        }
        self._logger.debug(self._format_entry(entry))

    def log_dify_response(
        self,
        request_id: str,
        status_code: int,
        body: Any,
        conversation_id: Optional[str] = None,
        is_stream_summary: bool = False,
    ) -> None:
        if not self._logger:
            return
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": "DIFY → PROXY",
            "stage": (
                "dify_response_stream_summary" if is_stream_summary else "dify_response"
            ),
            "request_id": request_id,
            "status_code": status_code,
            "conversation_id": conversation_id,
        }
        if isinstance(body, dict):
            entry["answer_length"] = len(body.get("answer", ""))
            entry["body"] = self._truncate_body(body)
        elif isinstance(body, str):
            entry["answer_length"] = len(body)
            entry["body"] = self._truncate_body(body)
        else:
            entry["body"] = str(body)[:2000] if body else None
        self._logger.debug(self._format_entry(entry))

    def log_openai_response(
        self,
        request_id: str,
        body: Dict[str, Any],
        conversation_id: Optional[str] = None,
    ) -> None:
        if not self._logger:
            return
        choices = body.get("choices", [])
        finish_reason = choices[0].get("finish_reason") if choices else None
        has_tool_calls = bool(
            choices
            and isinstance(choices[0], dict)
            and choices[0].get("message", {}).get("tool_calls")
        )
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": "PROXY → CLIENT",
            "stage": "openai_response",
            "request_id": request_id,
            "conversation_id": conversation_id,
            "finish_reason": finish_reason,
            "has_tool_calls": has_tool_calls,
            "usage": body.get("usage"),
            "body": self._truncate_body(body),
        }
        self._logger.debug(self._format_entry(entry))

    def log_stream_complete(
        self,
        request_id: str,
        accumulated_text: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        conversation_id: Optional[str],
        finish_reason: str,
    ) -> None:
        if not self._logger:
            return
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": "PROXY → CLIENT (stream summary)",
            "stage": "openai_stream_complete",
            "request_id": request_id,
            "conversation_id": conversation_id,
            "finish_reason": finish_reason,
            "accumulated_text_length": len(accumulated_text),
            "accumulated_text": self._truncate_body(accumulated_text),
            "tool_calls_count": len(tool_calls) if tool_calls else 0,
            "tool_calls": tool_calls,
        }
        self._logger.debug(self._format_entry(entry))

    def log_error(
        self, request_id: str, stage: str, error: str, status_code: Optional[int] = None
    ) -> None:
        if not self._logger:
            return
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": "ERROR",
            "stage": stage,
            "request_id": request_id,
            "status_code": status_code,
            "error": error,
        }
        self._logger.debug(self._format_entry(entry))


traffic_log = RequestResponseLogger()
