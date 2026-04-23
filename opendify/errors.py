"""错误类型与上游异常转换。"""

import json
from typing import Any, Dict, Optional

from .traffic_log import traffic_log


class APIError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        code: Optional[str] = None,
        param: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.code = code
        self.param = param

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


def raise_upstream_error(status_code: int, body: bytes, request_id: str = "") -> None:
    try:
        err = json.loads(body or b"{}")
        msg = err.get("message", body.decode("utf-8", errors="ignore"))
        code = err.get("code", "upstream_error")
    except Exception:
        msg = body.decode("utf-8", errors="ignore") if body else "Upstream error"
        code = "upstream_error"

    traffic_log.log_error(
        request_id, "dify_upstream", str(msg), status_code=status_code
    )

    if status_code == 429:
        etype = "rate_limit_error"
    elif status_code >= 500:
        etype = "server_error"
    else:
        etype = "invalid_request_error"
    raise APIError(status_code, str(msg), error_type=etype, code=str(code))
