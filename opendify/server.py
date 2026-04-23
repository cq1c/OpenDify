"""FastAPI 应用装配：lifespan、CORS、异常处理、v1 路由。"""

import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from . import http_client
from .auth import verify_api_key
from .config import (
    CONVERSATION_MODE,
    DIFY_API_BASE,
    MODEL_KEY_MAP,
    REQUEST_LOG_DIR,
    REQUEST_LOG_ENABLED,
    TIMEOUT,
    logger,
)
from .errors import APIError, raise_upstream_error
from .responses import build_openai_response
from .sessions import sessions
from .streaming import stream_and_capture_cid
from .traffic_log import traffic_log
from .transforms import extract_text, transform_openai_to_dify
from .utils import fast_id


@asynccontextmanager
async def lifespan(_: FastAPI):
    http_client.client = http_client.create_client()
    log_status = "开启" if REQUEST_LOG_ENABLED else "关闭"
    logger.info(
        "OpenDify Lite 启动 | 模型: %s | 会话模式: %s | 流量日志: %s",
        list(MODEL_KEY_MAP.keys()),
        CONVERSATION_MODE,
        log_status,
    )
    if REQUEST_LOG_ENABLED:
        logger.info("流量日志目录: %s", os.path.abspath(REQUEST_LOG_DIR))
    try:
        yield
    finally:
        if http_client.client is not None:
            await http_client.client.aclose()
            http_client.client = None


app = FastAPI(
    title="OpenDify Lite",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,
)


@app.exception_handler(APIError)
async def _api_error_handler(_: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


# ────────────────────────────────────────
#  POST /v1/chat/completions
# ────────────────────────────────────────


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, api_key: str = Depends(verify_api_key)):
    request_id = f"req_{secrets.token_hex(6)}"

    try:
        body = await request.body()
        openai_req = json.loads(body)
    except (json.JSONDecodeError, Exception):
        raise APIError(400, "Invalid JSON", code="invalid_json")

    model = openai_req.get("model")
    if not isinstance(model, str) or not model.strip():
        raise APIError(400, "Missing 'model'", code="missing_parameter", param="model")

    messages = openai_req.get("messages")
    if not isinstance(messages, list) or not messages:
        raise APIError(
            400, "Missing 'messages'", code="missing_parameter", param="messages"
        )

    # 记录 OpenAI 请求
    traffic_log.log_openai_request(request_id, model, openai_req)

    dify_key = MODEL_KEY_MAP.get(model)
    if not dify_key:
        if len(MODEL_KEY_MAP) == 1:
            dify_key = next(iter(MODEL_KEY_MAP.values()))
        else:
            raise APIError(
                404, f"Model '{model}' not found", code="model_not_found", param="model"
            )

    system_text = ""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            system_text = extract_text(m.get("content"))
            break

    session = sessions.get(model, system_text)

    if CONVERSATION_MODE == "auto":
        non_system_count = sum(
            1 for m in messages if isinstance(m, dict) and m.get("role") != "system"
        )
        is_new_task = sessions.maybe_reset_for_new_task(session, non_system_count)
        if is_new_task:
            logger.info(
                "检测到新任务（消息数从 >1 回落到 %d），重置会话 key=%s",
                non_system_count,
                session["key"],
            )
            # 新任务：丢弃历史 token，直接生成一个全新的
            session["token"] = secrets.token_hex(3)
            session["prev_tokens"] = []
        elif session["msg_count"] > 1:
            # 每次后续消息都轮换 token，提示词里会声明历史 token 全部作废
            sessions.rotate_token(session)
    else:
        # 非 conversation 模式：每次都是全新对话，Dify 看不到历史。
        # 历史 token 对模型毫无意义（它压根没见过），直接清空并换新 token。
        session["token"] = secrets.token_hex(3)
        session["prev_tokens"] = []
        session["conversation_id"] = None

    explicit_cid = request.headers.get("X-Dify-Conversation-Id") or openai_req.get(
        "conversation_id"
    )
    if isinstance(explicit_cid, str) and explicit_cid.strip():
        session["conversation_id"] = explicit_cid

    dify_req = transform_openai_to_dify(openai_req, session, dify_key=dify_key)
    if not dify_req:
        raise APIError(400, "Failed to transform request", code="transform_error")

    tool_token = dify_req.pop("_tool_token", None)

    # 记录 Dify 请求
    traffic_log.log_dify_request(
        request_id,
        f"{DIFY_API_BASE}/chat-messages",
        dify_req,
        conversation_id=session.get("conversation_id"),
    )

    headers = {
        "Authorization": f"Bearer {dify_key}",
        "Content-Type": "application/json",
    }
    endpoint = f"{DIFY_API_BASE}/chat-messages"
    stream = bool(openai_req.get("stream", False))

    client = http_client.client
    if client is None:
        raise APIError(500, "HTTP client not initialized", code="client_uninitialized")

    if stream:
        stream_opts = openai_req.get("stream_options") or {}
        include_usage = bool(
            stream_opts.get("include_usage") if isinstance(stream_opts, dict) else False
        )
        message_id = f"chatcmpl-{fast_id()}"

        cm = client.stream(
            "POST",
            endpoint,
            content=json.dumps(dify_req, ensure_ascii=False),
            headers=headers,
            timeout=TIMEOUT,
        )
        upstream = await cm.__aenter__()

        if upstream.status_code != 200:
            err_body = await upstream.aread()
            await cm.__aexit__(None, None, None)
            raise_upstream_error(upstream.status_code, err_body, request_id)

        return StreamingResponse(
            stream_and_capture_cid(
                upstream,
                model=model,
                message_id=message_id,
                tool_token=tool_token,
                include_usage=include_usage,
                session=session,
                request_id=request_id,
                tools=openai_req.get("tools") or None,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
            background=BackgroundTask(cm.__aexit__, None, None, None),
        )

    # ── 非流式 ──
    resp = await client.post(
        endpoint,
        content=json.dumps(dify_req, ensure_ascii=False),
        headers=headers,
        timeout=TIMEOUT,
    )

    if resp.status_code != 200:
        raise_upstream_error(resp.status_code, resp.content, request_id)

    dify_resp = json.loads(resp.content)

    # 记录 Dify 响应
    traffic_log.log_dify_response(
        request_id,
        resp.status_code,
        dify_resp,
        conversation_id=dify_resp.get("conversation_id"),
    )

    openai_resp, cid = build_openai_response(
        dify_resp,
        model,
        tool_token,
        session=session,
        tools=openai_req.get("tools") or None,
    )

    if cid:
        sessions.update_conversation_id(session, cid)

    # 记录 OpenAI 响应
    traffic_log.log_openai_response(request_id, openai_resp, conversation_id=cid)

    resp_headers: Dict[str, str] = {"Access-Control-Allow-Origin": "*"}
    if cid:
        resp_headers["X-Dify-Conversation-Id"] = cid
    return JSONResponse(content=openai_resp, headers=resp_headers)


# ────────────────────────────────────────
#  GET /v1/models
# ────────────────────────────────────────


@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    ts = int(time.time())
    data = [
        {"id": name, "object": "model", "created": ts, "owned_by": "dify"}
        for name in MODEL_KEY_MAP
    ]
    return JSONResponse(
        content={"object": "list", "data": data},
        headers={"Access-Control-Allow-Origin": "*"},
    )


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, api_key: str = Depends(verify_api_key)):
    if model_id not in MODEL_KEY_MAP:
        raise APIError(404, f"Model '{model_id}' not found", code="model_not_found")
    ts = int(time.time())
    return JSONResponse(
        content={"id": model_id, "object": "model", "created": ts, "owned_by": "dify"},
        headers={"Access-Control-Allow-Origin": "*"},
    )
