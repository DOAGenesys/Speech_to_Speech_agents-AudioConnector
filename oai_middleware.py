import asyncio
import hmac
import json
import uuid
import logging
import websockets
import http
import os
from typing import List

try:
    from websockets.asyncio.server import ServerConnection as _ServerConnection
    from websockets.http11 import Request as _WsRequest
except Exception:
    _ServerConnection = None
    _WsRequest = None

from config import (
    GENESYS_PATH,
    logger,
    LOG_FILE,
    DEBUG,
    GENESYS_API_KEY,
    AI_VENDOR,
    AI_MODEL,
)

from audio_hook_server import AudioHookServer
from utils import format_json
from datetime import datetime

async def validate_request(path_or_connection, headers_or_request):
    log_buffer: List[str] = []

    def _buffer_info(message: str):
        log_buffer.append(message)

    def _flush_buffer():
        for entry in log_buffer:
            logger.info(entry)
        log_buffer.clear()

    _buffer_info(f"\n{'='*50}\n[HTTP] Incoming request validation")
    connection = None
    request = None

    if _ServerConnection and isinstance(path_or_connection, _ServerConnection):
        connection = path_or_connection

    if _WsRequest and isinstance(headers_or_request, _WsRequest):
        request = headers_or_request
    elif connection and _WsRequest and isinstance(getattr(connection, "request", None), _WsRequest):
        request = connection.request

    if request:
        path_value = request.path
        header_source = request.headers
    else:
        path_value = path_or_connection
        header_source = headers_or_request

    if isinstance(path_value, str):
        request_path = path_value
    else:
        request_path = getattr(path_value, "path", None)
        if request_path is None:
            request_path = str(path_value)

    _buffer_info(f"[HTTP] Request path: {request_path}")
    _buffer_info(f"[HTTP] Expected WebSocket path: {GENESYS_PATH}")

    def build_header_map(source):
        pairs = None
        raw_items = getattr(source, 'raw_items', None)
        if callable(raw_items):
            pairs = list(raw_items())
        else:
            items_fn = getattr(source, 'items', None)
            if callable(items_fn):
                pairs = list(items_fn())
        if pairs is None:
            try:
                pairs = list(source)
            except Exception:
                pairs = []
        return {str(k).lower(): str(v) for k, v in pairs}

    header_keys = build_header_map(header_source or {})

    remote_address = None
    if connection is not None:
        remote_address = getattr(connection, "remote_address", None)
    if remote_address and isinstance(remote_address, (tuple, list)) and len(remote_address) >= 2:
        remote_repr = f"{remote_address[0]}:{remote_address[1]}"
    else:
        remote_repr = header_keys.get('host', 'unknown')

    _buffer_info(f"[HTTP] Remote address: {remote_repr}")

    header_lines: List[str] = ["[HTTP] Full headers received:"]
    for name, value in header_keys.items():
        if name in ['x-api-key', 'authorization']:
            header_lines.append(f"[HTTP]   {name}: {'*' * 8}")
        else:
            header_lines.append(f"[HTTP]   {name}: {value}")
    for line in header_lines:
        _buffer_info(line)

    upgrade_header = header_keys.get('upgrade', '').lower()

    def _build_response(status: http.HTTPStatus, text: str):
        if connection and hasattr(connection, "respond"):
            return connection.respond(status, text)
        return status, [], text.encode()

    if request_path == '/' or request_path == '':
        if upgrade_header != 'websocket':
            logger.debug("[HTTP] Health check request detected at root path, returning 200 OK")
            return _build_response(http.HTTPStatus.OK, 'OK\n')

    _flush_buffer()

    if not request_path.startswith(GENESYS_PATH):
        logger.error("[HTTP] Path mismatch:")
        logger.error(f"[HTTP]   Expected path to start with: {GENESYS_PATH}")
        logger.error(f"[HTTP]   Received path: {request_path}")
        return _build_response(http.HTTPStatus.NOT_FOUND, 'Invalid path\n')
    
    logger.info(f"[HTTP] Path validation passed: {request_path} matches {GENESYS_PATH}")

    # --- Start of Security Update ---
    # Check for the presence and value of the x-api-key
    incoming_api_key = header_keys.get('x-api-key')

    if not incoming_api_key:
        logger.error("[HTTP] Connection rejected - Missing 'x-api-key' header.")
        return _build_response(http.HTTPStatus.UNAUTHORIZED, "Missing 'x-api-key' header\n")

    if not hmac.compare_digest(incoming_api_key.encode("utf-8"), GENESYS_API_KEY.encode("utf-8")):
        logger.error("[HTTP] Connection rejected - Invalid API Key.")
        return _build_response(http.HTTPStatus.UNAUTHORIZED, "Invalid API Key\n")

    logger.info("[HTTP] API Key validation successful.")
    # --- End of Security Update ---


    required_headers = [
        'audiohook-organization-id',
        'audiohook-correlation-id',
        'audiohook-session-id',
        'upgrade',
        'sec-websocket-version',
        'sec-websocket-key'
    ]

    missing_headers = []
    found_headers = []
    for h in required_headers:
        if h.lower() not in header_keys:
            missing_headers.append(h)
        else:
            found_headers.append(h)

    if missing_headers:
        error_msg = f"Missing required headers (excluding x-api-key): {', '.join(missing_headers)}"
        logger.error(f"[HTTP] Connection rejected - {error_msg}")
        logger.error("[HTTP] Found headers: " + ", ".join(found_headers))
        return _build_response(http.HTTPStatus.BAD_REQUEST, error_msg)

    upgrade_header = header_keys.get('upgrade', '').lower()
    logger.info(f"[HTTP] Checking upgrade header: {upgrade_header}")
    if upgrade_header != 'websocket':
        error_msg = f"Invalid upgrade header: {upgrade_header}"
        logger.error(f"[HTTP] {error_msg}")
        return _build_response(http.HTTPStatus.BAD_REQUEST, 'WebSocket upgrade required\n')

    ws_version = header_keys.get('sec-websocket-version', '')
    logger.info(f"[HTTP] Checking WebSocket version: {ws_version}")
    if ws_version != '13':
        error_msg = f"Invalid WebSocket version: {ws_version}"
        logger.error(f"[HTTP] {error_msg}")
        return _build_response(http.HTTPStatus.BAD_REQUEST, 'WebSocket version 13 required\n')

    ws_key = header_keys.get('sec-websocket-key')
    if not ws_key:
        logger.error("[HTTP] Missing WebSocket key")
        return _build_response(http.HTTPStatus.BAD_REQUEST, 'WebSocket key required\n')
    logger.info("[HTTP] Found valid WebSocket key")

    ws_protocol = header_keys.get('sec-websocket-protocol', '')
    if ws_protocol:
        logger.info(f"[HTTP] WebSocket protocol requested: {ws_protocol}")
        if 'audiohook' not in ws_protocol.lower():
            logger.warning("[HTTP] Client didn't request 'audiohook' protocol")

    connection_header = header_keys.get('connection', '').lower()
    logger.info(f"[HTTP] Connection header: {connection_header}")
    if 'upgrade' not in connection_header:
        logger.warning("[HTTP] Connection header doesn't contain 'upgrade'")

    logger.info("[HTTP] All validation checks passed successfully")
    logger.info(f"[HTTP] Proceeding with WebSocket upgrade")
    logger.info("="*50)
    return None

async def handle_genesys_connection(websocket):
    connection_id = str(uuid.uuid4())[:8]
    logger.info(f"\n{'='*50}\n[WS-{connection_id}] New WebSocket connection handler started")

    session = None

    try:
        logger.info(f"Received WebSocket connection from {websocket.remote_address}")
        logger.info(f"[WS-{connection_id}] Remote address: {websocket.remote_address}")
        logger.info(f"[WS-{connection_id}] Connection state: {websocket.state}")

        ws_attributes = ['path', 'remote_address', 'local_address', 'state', 'open', 'protocol']
        logger.info(f"[WS-{connection_id}] WebSocket object attributes:")
        for attr in ws_attributes:
            value = getattr(websocket, attr, "Not available")
            logger.info(f"[WS-{connection_id}]   {attr}: {value}")

        logger.info(f"[WS-{connection_id}] WebSocket connection established; handshake was validated beforehand.")

        session = AudioHookServer(websocket)
        logger.info(f"[WS-{connection_id}] Session created with ID: {session.session_id}")

        logger.info(f"[WS-{connection_id}] Starting main message loop")
        while session.running:
            try:
                logger.debug(f"[WS-{connection_id}] Waiting for next message...")
                msg = await websocket.recv()
                if isinstance(msg, bytes):
                    logger.debug(f"[WS-{connection_id}] Received binary frame: {len(msg)} bytes")
                    await session.handle_audio_frame(msg)
                else:
                    try:
                        data = json.loads(msg)
                        logger.debug(f"[WS-{connection_id}] Received JSON message:\n{format_json(data)}")
                        await session.handle_message(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"[WS-{connection_id}] Error parsing JSON: {e}")
                        await session.disconnect_session("error", f"JSON parse error: {e}")
                    except Exception as e:
                        logger.error(f"[WS-{connection_id}] Error processing message: {e}")
                        await session.disconnect_session("error", f"Message processing error: {e}")

            except websockets.ConnectionClosed as e:
                close_frame = getattr(e, "rcvd", None)
                close_code = close_frame.code if close_frame else None
                close_reason = close_frame.reason if close_frame else ""
                logger.info(f"[WS-{connection_id}] Connection closed: code={close_code}, reason={close_reason}")
                break
            except Exception as e:
                logger.error(f"[WS-{connection_id}] Unexpected error: {e}", exc_info=True)
                break

        logger.info(f"[WS-{connection_id}] Session loop ended, cleaning up")
        if session and session.ai_client:
            await session.ai_client.close()
        logger.info(f"[WS-{connection_id}] Session cleanup complete")

    except Exception as e:
        logger.error(f"[WS-{connection_id}] Fatal connection error: {e}", exc_info=True)
        if session is None:
            session = AudioHookServer(websocket)
        await session.disconnect_session(reason="error", info=f"Internal error: {str(e)}")
    finally:
        logger.info(f"[WS-{connection_id}] Connection handler finished\n{'='*50}")

async def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))
    
    startup_msg = f"""
{'='*80}
Genesys AudioHook Speech-to-Speech Bridge (OpenAI / Gemini / xAI Grok)
Default vendor: {AI_VENDOR} ({AI_MODEL})
Starting up at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Host: {host}
Port: {port}
Path: {GENESYS_PATH}
SSL: Managed by deployment platform
Log File: {os.path.abspath(LOG_FILE)}
{'='*80}
"""
    logger.info(startup_msg)

    websockets_logger = logging.getLogger('websockets')
    if DEBUG != 'true':
        websockets_logger.setLevel(logging.INFO)

    websockets_logger.addHandler(logging.FileHandler(LOG_FILE))

    try:
        async with websockets.serve(
            handle_genesys_connection,
            host,
            port,
            process_request=validate_request, # Use the updated validation function
            max_size=64000,
            ping_interval=None,
            ping_timeout=None
        ):
            logger.info(
                f"Server is listening for Genesys AudioHook connections on "
                f"ws://{host}:{port}{GENESYS_PATH}"
            )
            
            try:
                await asyncio.Future()  # run forever
            except asyncio.CancelledError:
                logger.info("Server shutdown initiated")
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down via KeyboardInterrupt.")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Server shutdown complete.")
