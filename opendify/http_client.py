"""共享的 HTTP 客户端 (httpx AsyncClient)。"""

from typing import Optional

import httpx

from .config import POOL_SIZE, TIMEOUT

# 由 lifespan 在启动时赋值，关闭时清理。
client: Optional[httpx.AsyncClient] = None


def create_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=TIMEOUT, connect=10.0),
        limits=httpx.Limits(
            max_keepalive_connections=POOL_SIZE,
            max_connections=POOL_SIZE,
            keepalive_expiry=30.0,
        ),
        http2=True,
    )
