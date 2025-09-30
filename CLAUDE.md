# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenDify is a high-performance FastAPI proxy service that converts Dify API calls into OpenAI-compatible format. This allows developers to use OpenAI-style SDKs and clients to access Dify applications without modifying existing OpenAI-based code.

## Development Commands

### Setup and Installation
```bash
pip install -r requirements.txt
```

### Running the Application

**Development Mode** (with auto-reload):
```bash
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

**Production Mode**:
```bash
python app.py
```

### Testing
```bash
python test.py
```
This runs a basic test against the local proxy server using the OpenAI client library.

## Environment Configuration

Create a `.env` file based on `.env.example`:
- `DIFY_API_KEYS`: Comma-separated Dify API keys for the applications you want to proxy
- `DIFY_API_BASE`: Base URL for Dify API (e.g., `https://api.dify.ai/v1`)
- `VALID_API_KEYS`: Comma-separated API keys allowed to access the proxy
- `CONVERSATION_MEMORY_MODE`: Either `1` (history messages mode) or `2` (conversation_id mode)
- `SERVER_HOST`: Server host (default: `127.0.0.1`)
- `SERVER_PORT`: Server port (default: `8000`)
- `TIMEOUT`: Request timeout in seconds (default: `30.0`)
- `WORKERS`: Number of uvicorn workers for production (default: `1`)

## Architecture

### Core Components

**app.py** - Main application file containing:
- `DifyModelManager`: Manages Dify model to API key mappings with caching (30-minute TTL)
- HTTP/2 client with connection pooling (100 connections max)
- API endpoints: `/v1/chat/completions` and `/v1/models`

**utils/tools.py** - Tool processing utilities:
- `generate_tool_prompt()`: Injects tool definitions into system prompts
- `process_messages_with_tools()`: Handles message preprocessing for tool support
- `extract_tool_invocations()`: Extracts tool calls from model responses using multiple strategies (JSON blocks, inline JSON with bracket balancing, natural language patterns)
- `remove_tool_json_content()`: Cleans tool JSON from response text

**utils/tool_handler.py** - SSE tool handler:
- `SSEToolHandler`: Manages streaming tool call responses
- Uses `edit_index` and `edit_content` mechanism for incremental content updates
- Supports partial tool block parsing with bracket-balanced JSON extraction
- Handles both complete and incomplete tool invocations during streaming

### Key Design Patterns

**Conversation Memory Modes**:
1. **Mode 1 (History Messages)**: Appends conversation history to the current query as structured text
2. **Mode 2 (Conversation ID)**: Uses zero-width character encoding to embed `conversation_id` in assistant responses for stateful conversations

**Tool Call Support**:
- Tool definitions are injected into system prompts with structured markdown format
- Model responses are scanned (up to `SCAN_LIMIT=8000` chars) for tool invocations
- Three extraction strategies: fenced JSON blocks, bracket-balanced inline JSON, natural language patterns
- Tool calls are converted to OpenAI-compatible format with proper `id`, `type`, and `function` fields

**Zero-Width Character Encoding** (Mode 2):
- `encode_conversation_id()`: Base64 encodes conversation IDs with zero-width chars every 4 characters
- `decode_conversation_id()`: Extracts and decodes conversation IDs from response text
- Makes conversation IDs invisible to end users while preserving state

**Performance Optimizations**:
- HTTP/2 support with connection pooling
- LRU caching for conversation ID encoding (256 entries)
- 30-minute TTL cache for app info to reduce API calls
- ujson for faster JSON parsing
- Async/await throughout for non-blocking I/O
- Simplified logging (WARNING level, ERROR for third-party libs)

### API Transformation Flow

1. **Request arrives** at `/v1/chat/completions` with OpenAI format
2. **Model lookup**: Maps OpenAI model name to Dify API key via cached app info
3. **Message processing**:
   - Extracts system prompts and user queries
   - Injects tool definitions if tools are provided
   - Handles tool/function messages by converting to assistant format
   - Applies conversation memory mode logic
4. **Dify API call**: Transforms to Dify format with `inputs`, `query`, `response_mode`, `conversation_id` (if mode 2), `user`
5. **Response handling**:
   - **Streaming**: Processes SSE events (`message`, `agent_message`, `agent_thought`, `tool_calls`, `message_end`)
   - **Non-streaming**: Extracts tool calls, removes tool JSON from content, encodes conversation ID
6. **Response transformation**: Converts Dify response to OpenAI format with proper `choices`, `finish_reason`, `tool_calls`

### Tool Call Processing Details

**Extraction** (app.py:411-499):
- Scans response text up to `SCAN_LIMIT` (8000 chars)
- Pattern 1: `TOOL_CALL_FENCE_PATTERN` for ```json blocks
- Pattern 2: Bracket-balanced JSON object extraction (handles nested braces, string escaping)
- Pattern 3: `FUNCTION_CALL_PATTERN` for natural language like "调用函数: func_name 参数: {...}"
- Ensures `arguments` field is always a JSON string, not an object

**Removal** (app.py:502-556):
- First pass: Removes fenced JSON blocks containing `tool_calls`
- Second pass: Uses bracket balancing to find and remove inline tool JSON
- Preserves all other content including non-tool JSON objects

**Streaming** (app.py:631-768):
- Accumulates content in `accumulated_content` buffer
- Continuously checks for tool invocations using `extract_tool_invocations()`
- Once detected, switches to `tool_calls_sent=True` mode
- Sends role chunk, then tool_calls delta, then final chunk with `finish_reason: "tool_calls"`
- Handles both model-generated tool JSON and native Dify `tool_calls` events

## Important Notes

- The `DifyModelManager` fetches app info from Dify's `/info` endpoint to map app names to API keys
- All responses include CORS headers (`*`) for cross-origin access
- The service uses `ujson` instead of standard `json` for better performance
- Logging is minimal in production (WARNING level) to reduce overhead
- Windows systems use `asyncio` event loop; Unix systems use `uvloop` for better performance