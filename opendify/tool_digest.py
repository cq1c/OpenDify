"""
工具描述摘要缓存（仅针对 function.description；参数 schema 永不压缩）：
  - 首次见到某工具（按 name+description hash）→ 主流程用原始描述，异步后台调 Dify 生成摘要
  - 后台完成后写入 TOOL_DESC_DIGEST_DIR/<hash>.txt
  - 下次命中 → 直接用极短摘要替换原描述，节省上下文
后台任务失败静默退化，主请求永不因此阻塞。
"""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

from . import http_client
from .config import (
    DIFY_API_BASE,
    TIMEOUT,
    TOOL_DESC_DIGEST_DIR,
    TOOL_DESC_DIGEST_ENABLED,
    TOOL_DESC_DIGEST_MAX_CHARS,
    logger,
)


class ToolDescDigestCache:
    def __init__(self) -> None:
        self._dir = Path(TOOL_DESC_DIGEST_DIR)
        if TOOL_DESC_DIGEST_ENABLED:
            self._dir.mkdir(parents=True, exist_ok=True)
        self._mem: Dict[str, str] = {}
        self._pending: set = set()

    @staticmethod
    def _hash(name: str, desc: str) -> str:
        return hashlib.md5(f"{name}|{desc}".encode("utf-8")).hexdigest()[:16]

    def _path(self, h: str) -> Path:
        return self._dir / f"{h}.txt"

    def load(self, name: str, desc: str) -> Optional[str]:
        if not TOOL_DESC_DIGEST_ENABLED or not desc:
            return None
        h = self._hash(name, desc)
        if h in self._mem:
            return self._mem[h]
        p = self._path(h)
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8").strip()
                if text:
                    self._mem[h] = text
                    return text
            except Exception:
                return None
        return None

    def schedule_generate(self, dify_key: str, name: str, desc: str) -> None:
        if not TOOL_DESC_DIGEST_ENABLED or not desc or not dify_key:
            return
        h = self._hash(name, desc)
        if h in self._mem or h in self._pending or self._path(h).exists():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._pending.add(h)
        loop.create_task(self._generate(h, dify_key, name, desc))

    async def _generate(self, h: str, dify_key: str, name: str, desc: str) -> None:
        try:
            limit = TOOL_DESC_DIGEST_MAX_CHARS
            prompt = (
                f"把下面这个工具的描述压缩为一行、不超过 {limit} 字的极简说明，"
                "只保留核心用途与关键参数提示。严禁寒暄、前缀、引号与标点包装，"
                "直接输出压缩后的纯文本：\n\n"
                f"工具名: {name}\n描述: {desc}"
            )
            body = {
                "inputs": {},
                "query": prompt,
                "response_mode": "blocking",
                "user": "opendify_digest",
            }
            client = http_client.client
            if client is None:
                return
            rsp = await client.post(
                f"{DIFY_API_BASE}/chat-messages",
                content=json.dumps(body, ensure_ascii=False),
                headers={
                    "Authorization": f"Bearer {dify_key}",
                    "Content-Type": "application/json",
                },
                timeout=TIMEOUT,
            )
            if rsp.status_code != 200:
                logger.debug("digest http %s for %s", rsp.status_code, name)
                return
            data = json.loads(rsp.content)
            answer = (data.get("answer") or "").strip().split("\n", 1)[0].strip()
            if not answer:
                return
            hard_cap = max(limit * 3, 32)
            if len(answer) > hard_cap:
                answer = answer[:hard_cap]
            self._mem[h] = answer
            try:
                self._path(h).write_text(answer, encoding="utf-8")
            except Exception as e:
                logger.debug("digest write failed: %s", e)
        except Exception as e:
            logger.debug("digest task error: %s", e)
        finally:
            self._pending.discard(h)


tool_desc_digest = ToolDescDigestCache()
