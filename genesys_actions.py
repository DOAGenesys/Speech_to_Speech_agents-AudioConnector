import asyncio
import base64
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

from config import (
    GENESYS_ALLOWED_DATA_ACTION_IDS,
    GENESYS_BASE_URL,
    GENESYS_CLIENT_ID,
    GENESYS_CLIENT_SECRET,
    GENESYS_HTTP_RETRY_BACKOFF_SECONDS,
    GENESYS_HTTP_RETRY_MAX,
    GENESYS_HTTP_TIMEOUT_SECONDS,
    GENESYS_LOGIN_URL,
    GENESYS_MAX_ACTION_CALLS_PER_SESSION,
    GENESYS_MAX_TOOL_ARGUMENT_BYTES,
    GENESYS_MAX_TOOLS_PER_SESSION,
    GENESYS_REGION,
    GENESYS_TOKEN_CACHE_TTL_SECONDS,
    GENESYS_TOOL_OUTPUT_REDACTION_FIELDS,
    GENESYS_TOOLS_STRICT_MODE,
    logger,
)


class GenesysToolError(Exception):
    """Raised when a Genesys data action cannot be prepared or executed."""


def _derive_api_base_url() -> str:
    if GENESYS_BASE_URL:
        return GENESYS_BASE_URL.rstrip('/')
    if GENESYS_REGION:
        if 'mypurecloud.com' in GENESYS_REGION or 'mypurecloud.de' in GENESYS_REGION:
            return f"https://api.{GENESYS_REGION}"
        return f"https://api.{GENESYS_REGION}.mypurecloud.com"
    return "https://api.mypurecloud.com"


def _derive_login_url() -> str:
    if GENESYS_LOGIN_URL:
        return GENESYS_LOGIN_URL.rstrip('/')
    if GENESYS_REGION:
        if 'mypurecloud.com' in GENESYS_REGION or 'mypurecloud.de' in GENESYS_REGION:
            return f"https://login.{GENESYS_REGION}"
        return f"https://login.{GENESYS_REGION}.mypurecloud.com"
    return "https://login.mypurecloud.com"


def _sanitize_function_name(action_id: str) -> str:
    sanitized = ''.join(ch.lower() if ch.isalnum() else '_' for ch in action_id)
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')
    sanitized = sanitized.strip('_')
    if not sanitized:
        sanitized = 'action'
    if not sanitized[0].isalpha():
        sanitized = f"a_{sanitized}"
    return sanitized[:60]


def _normalize_parameters_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        **(schema or {}),
        "type": "object",
        "properties": (schema or {}).get("properties", {}) or {}
    }

    # Remove potentially problematic fields that could conflict with OpenAI's expectations
    base.pop("title", None)
    base.pop("$schema", None)
    
    # Ensure additionalProperties is set to false for consistency with OpenAI Realtime API
    # This prevents issues where Genesys schemas might have additionalProperties: true
    base["additionalProperties"] = False
    # "strict" is a tool-level flag in some vendor APIs, not a JSON Schema keyword.
    base.pop("strict", None)

    if not GENESYS_TOOLS_STRICT_MODE:
        # Even without strict mode, we need consistent schema format
        # Just don't enforce required fields on all nested properties
        return base

    # When strict mode is enabled, recursively enforce strict schema rules
    def enforce(obj: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(obj, dict):
            return obj
        copy = json.loads(json.dumps(obj))
        if copy.get("type") == "object":
            copy.setdefault("properties", {})
            copy["additionalProperties"] = False
            child_keys = list(copy["properties"].keys())
            if child_keys:
                copy["required"] = child_keys
            for key, value in copy["properties"].items():
                copy["properties"][key] = enforce(value)
        if copy.get("type") == "array" and "items" in copy:
            copy["items"] = enforce(copy["items"])
        return copy

    return enforce(base)


def _build_tool_description(action_id: str, schema: Dict[str, Any], custom: Optional[str]) -> str:
    if custom:
        return custom
    properties = schema.get("properties") or {}
    if not properties:
        return f"Executes the Genesys Cloud Data Action {action_id}."
    parts = []
    for key, prop in properties.items():
        desc = prop.get("description") if isinstance(prop, dict) else None
        if desc:
            parts.append(f"{key}: {desc}")
        else:
            parts.append(key)
    summary = '; '.join(parts)
    return f"Executes the Genesys Cloud Data Action {action_id}. Input fields: {summary}"


def _redact_payload(payload: Any) -> Any:
    if not GENESYS_TOOL_OUTPUT_REDACTION_FIELDS:
        return payload

    try:
        clone = json.loads(json.dumps(payload))
    except (TypeError, ValueError):
        return payload

    for path in GENESYS_TOOL_OUTPUT_REDACTION_FIELDS:
        segments = path.split('.')
        cursor = clone
        for segment in segments[:-1]:
            if isinstance(cursor, dict):
                cursor = cursor.get(segment)
            else:
                cursor = None
            if cursor is None:
                break
        if cursor is None:
            continue
        leaf = segments[-1]
        if isinstance(cursor, dict) and leaf in cursor:
            cursor[leaf] = "[REDACTED]"
    return clone


class GenesysOAuthClient:
    def __init__(self) -> None:
        self._token: Optional[Dict[str, Any]] = None
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        async with self._lock:
            if self._token and self._token.get('expires_at', 0) - 60 > time.time():
                return self._token['access_token']
            self._token = await self._fetch_token()
            return self._token['access_token']

    async def _fetch_token(self) -> Dict[str, Any]:
        if not GENESYS_CLIENT_ID or not GENESYS_CLIENT_SECRET:
            raise GenesysToolError("Genesys client credentials are not configured")

        login_url = _derive_login_url()
        auth_value = base64.b64encode(f"{GENESYS_CLIENT_ID}:{GENESYS_CLIENT_SECRET}".encode("utf-8")).decode("utf-8")
        data = {'grant_type': 'client_credentials'}

        attempt = 0
        while attempt <= GENESYS_HTTP_RETRY_MAX:
            attempt += 1
            try:
                async with httpx.AsyncClient(timeout=GENESYS_HTTP_TIMEOUT_SECONDS) as client:
                    response = await client.post(
                        f"{login_url}/oauth/token",
                        headers={
                            "Authorization": f"Basic {auth_value}",
                            "Content-Type": "application/x-www-form-urlencoded"
                        },
                        data=data,
                    )
                if response.status_code == 429:
                    retry_after = float(response.headers.get('Retry-After', '1'))
                    logger.warning(
                        f"[GenesysOAuth] Rate limited. Retry in {retry_after}s"
                    )
                    await asyncio.sleep(retry_after)
                    continue
                if response.status_code in (401, 403):
                    raise GenesysToolError("Genesys OAuth credentials rejected")
                response.raise_for_status()
                payload = response.json()
                expires_in = int(payload.get('expires_in', 3600))
                ttl = min(expires_in, GENESYS_TOKEN_CACHE_TTL_SECONDS)
                logger.info("[GenesysOAuth] Access token obtained")
                return {
                    'access_token': payload['access_token'],
                    'expires_at': time.time() + ttl
                }
            except (httpx.HTTPError, ValueError) as exc:
                if attempt > GENESYS_HTTP_RETRY_MAX:
                    raise GenesysToolError(f"Genesys OAuth failed: {exc}") from exc
                backoff = GENESYS_HTTP_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(f"[GenesysOAuth] Error during token fetch: {exc}. Backing off for {backoff:.2f}s")
                await asyncio.sleep(backoff)

        raise GenesysToolError("Unable to fetch Genesys OAuth token after retries")


class GenesysActionsClient:
    def __init__(self) -> None:
        self._oauth = GenesysOAuthClient()
        self._base_url = _derive_api_base_url()
        self._schema_cache: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}
        self._cache_lock = asyncio.Lock()

    async def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            token = await self._oauth.get_token()
        except Exception as token_err:
            logger.error(f"[GenesysActions] ERROR: Failed to get OAuth token: {token_err}", exc_info=True)
            raise GenesysToolError(f"Authentication failed: {str(token_err)}")
        
        attempt = 0
        url = f"{self._base_url}{path}"
        last_error = None

        while attempt <= GENESYS_HTTP_RETRY_MAX:
            attempt += 1
            try:
                async with httpx.AsyncClient(timeout=GENESYS_HTTP_TIMEOUT_SECONDS) as client:
                    response = await client.request(
                        method=method,
                        url=url,
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Content-Type": "application/json"
                        },
                        json=payload,
                    )
                
                if response.status_code == 429:
                    retry_after = float(response.headers.get('Retry-After', '1'))
                    logger.warning(
                        f"[GenesysActions] Rate limited on {method} {path} (attempt {attempt}/{GENESYS_HTTP_RETRY_MAX}). "
                        f"Retrying in {retry_after}s"
                    )
                    await asyncio.sleep(retry_after)
                    continue
                
                if response.status_code >= 500:
                    backoff = GENESYS_HTTP_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                    logger.error(
                        f"[GenesysActions] ERROR: Server error {response.status_code} on {method} {path} "
                        f"(attempt {attempt}/{GENESYS_HTTP_RETRY_MAX}). Backing off {backoff:.2f}s"
                    )
                    await asyncio.sleep(backoff)
                    continue
                
                if response.status_code >= 400:
                    # Error bodies of data actions can echo customer data, so only identifiers are logged.
                    try:
                        error_body = response.json()
                    except Exception:
                        error_body = {}
                    if not isinstance(error_body, dict):
                        error_body = {}
                    logger.error(
                        f"[GenesysActions] ERROR: HTTP {response.status_code} on {method} {path} "
                        f"(code={error_body.get('code')}, contextId={error_body.get('contextId')})"
                    )
                
                response.raise_for_status()
                
                try:
                    return response.json()
                except json.JSONDecodeError as json_err:
                    logger.error(
                        f"[GenesysActions] ERROR: Invalid JSON response from {method} {path} at position {json_err.pos}"
                    )
                    raise GenesysToolError("Invalid JSON response from Genesys API")
                    
            except httpx.TimeoutException as timeout_exc:
                last_error = timeout_exc
                logger.error(
                    f"[GenesysActions] ERROR: Timeout on {method} {path} (attempt {attempt}/{GENESYS_HTTP_RETRY_MAX}). "
                    f"Timeout: {GENESYS_HTTP_TIMEOUT_SECONDS}s"
                )
                if attempt > GENESYS_HTTP_RETRY_MAX:
                    raise GenesysToolError(f"Request timeout after {GENESYS_HTTP_RETRY_MAX} attempts") from timeout_exc
                backoff = GENESYS_HTTP_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                await asyncio.sleep(backoff)
            except httpx.HTTPStatusError as status_exc:
                last_error = status_exc
                logger.error(
                    f"[GenesysActions] ERROR: HTTP status error on {method} {path}: {status_exc}",
                    exc_info=True
                )
                if attempt > GENESYS_HTTP_RETRY_MAX:
                    raise GenesysToolError(f"HTTP {status_exc.response.status_code}: {str(status_exc)}") from status_exc
                backoff = GENESYS_HTTP_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                await asyncio.sleep(backoff)
            except httpx.HTTPError as http_exc:
                last_error = http_exc
                logger.error(
                    f"[GenesysActions] ERROR: HTTP error on {method} {path} (attempt {attempt}/{GENESYS_HTTP_RETRY_MAX}): {http_exc}",
                    exc_info=True
                )
                if attempt > GENESYS_HTTP_RETRY_MAX:
                    raise GenesysToolError(f"Genesys API request failed: {str(http_exc)}") from http_exc
                backoff = GENESYS_HTTP_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                await asyncio.sleep(backoff)

        error_msg = f"Genesys API request exhausted {GENESYS_HTTP_RETRY_MAX} retries for {method} {path}"
        if last_error:
            error_msg += f". Last error: {str(last_error)}"
        logger.error(f"[GenesysActions] CRITICAL ERROR: {error_msg}")
        raise GenesysToolError(error_msg)

    async def _get_schema(self, action_id: str, schema_type: str) -> Dict[str, Any]:
        cache_key = (action_id, schema_type)
        async with self._cache_lock:
            cached = self._schema_cache.get(cache_key)
            if cached and cached[0] > time.time():
                return cached[1]

        schema = await self._request(
            "GET",
            f"/api/v2/integrations/actions/{action_id}/schemas/{schema_type}schema.json"
        )
        normalized = {
            "type": "object",
            "properties": {},
            **schema
        }
        async with self._cache_lock:
            self._schema_cache[cache_key] = (time.time() + 600, normalized)
        return normalized

    async def get_input_schema(self, action_id: str) -> Dict[str, Any]:
        return await self._get_schema(action_id, "input")

    async def get_success_schema(self, action_id: str) -> Dict[str, Any]:
        return await self._get_schema(action_id, "success")

    async def execute(self, action_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/v2/integrations/actions/{action_id}/test",
            payload
        )


GENESYS_ACTIONS_CLIENT = GenesysActionsClient()


@dataclass
class GenesysToolContext:
    tools: List[Dict[str, Any]]
    instructions: Optional[str]
    handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]]


def _parse_action_ids(raw_value: Optional[str]) -> List[str]:
    if not raw_value:
        return []
    normalized = raw_value.replace('|', ',').replace('\n', ',').replace(';', ',')
    return [token.strip() for token in normalized.split(',') if token.strip()]


def _parse_descriptions(raw_value: Optional[str], action_ids: List[str]) -> Dict[str, str]:
    if not raw_value:
        return {}
    segments = [seg.strip() for seg in raw_value.split('|') if seg.strip()]
    if len(segments) != len(action_ids):
        logger.warning(
            "[GenesysTools] data_action_descriptions count (%s) does not match ids count (%s)",
            len(segments),
            len(action_ids)
        )
        return {}
    return {action_ids[idx]: segments[idx] for idx in range(len(action_ids))}


def _build_instruction_text(tool_summaries: List[Tuple[str, str, List[str]]]) -> str:
    if not tool_summaries:
        return ""
    lines = [
        "Genesys Cloud data action tools are available in this call.",
        "Always invoke the relevant tool to retrieve or update Genesys data instead of guessing values.",
        "After you receive the tool result, interpret the JSON payload and explain it to the caller in plain language.",
        "Available tools:" 
    ]
    for name, action_id, params in tool_summaries:
        param_text = ', '.join(params) if params else 'no parameters'
        lines.append(f"- {name} (action {action_id}) parameters: {param_text}")
    return '\n'.join(lines)


async def build_genesys_tool_context(session_logger, input_variables: Dict[str, Any]) -> Optional[GenesysToolContext]:
    raw_ids = input_variables.get("DATA_ACTION_IDS") or input_variables.get("GENESYS_DATA_ACTION_IDS")
    action_ids = _parse_action_ids(raw_ids)
    if not action_ids:
        return None

    if not GENESYS_CLIENT_ID or not GENESYS_CLIENT_SECRET:
        session_logger.error("[GenesysTools] Data actions requested but GENESYS_CLIENT_ID / GENESYS_CLIENT_SECRET are not configured")
        return None

    if GENESYS_ALLOWED_DATA_ACTION_IDS:
        before = len(action_ids)
        action_ids = [aid for aid in action_ids if aid in GENESYS_ALLOWED_DATA_ACTION_IDS]
        if before != len(action_ids):
            session_logger.warning(
                "[GenesysTools] Filtered %s action ids not on allowlist",
                before - len(action_ids)
            )
        if not action_ids:
            session_logger.warning("[GenesysTools] No requested data actions allowed for this session")
            return None

    action_ids = action_ids[:GENESYS_MAX_TOOLS_PER_SESSION]
    description_map = _parse_descriptions(
        input_variables.get("DATA_ACTION_DESCRIPTIONS"),
        action_ids
    )

    session_logger.info(
        f"[GenesysTools] Preparing {len(action_ids)} data action tools"
    )

    schema_tasks = [
        asyncio.gather(
            GENESYS_ACTIONS_CLIENT.get_input_schema(action_id),
            GENESYS_ACTIONS_CLIENT.get_success_schema(action_id),
            return_exceptions=False
        )
        for action_id in action_ids
    ]

    tools: List[Dict[str, Any]] = []
    tool_summaries: List[Tuple[str, str, List[str]]] = []
    handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = {}
    invocation_counter = {'count': 0}

    results = await asyncio.gather(*schema_tasks, return_exceptions=True)
    for action_id, result in zip(action_ids, results):
        if isinstance(result, Exception):
            session_logger.error(
                f"[GenesysTools] Failed to fetch schema for {action_id}: {result}"
            )
            continue

        input_schema, success_schema = result
        base_tool_name = f"genesys_data_action_{_sanitize_function_name(action_id)}"
        tool_name = base_tool_name
        suffix = 2
        while tool_name in handlers:
            tool_name = f"{base_tool_name}_{suffix}"
            suffix += 1
        parameters = _normalize_parameters_schema(input_schema)
        description = _build_tool_description(action_id, input_schema, description_map.get(action_id))
        tool_def = {
            "type": "function",
            "name": tool_name,
            "description": description,
            "parameters": parameters
        }
        tools.append(tool_def)
        param_keys = list(parameters.get("properties", {}).keys())
        tool_summaries.append((tool_name, action_id, param_keys))

        async def handler(args: Dict[str, Any], *, _action_id=action_id, _success_schema=success_schema) -> Dict[str, Any]:
            try:
                if invocation_counter['count'] >= GENESYS_MAX_ACTION_CALLS_PER_SESSION:
                    session_logger.error(
                        "[GenesysTools] ERROR: Maximum invocations (%s) exceeded for session",
                        GENESYS_MAX_ACTION_CALLS_PER_SESSION
                    )
                    raise GenesysToolError("Maximum Genesys data action invocations exceeded for this session")
                
                if not isinstance(args, dict):
                    session_logger.error(
                        "[GenesysTools] ERROR: Invalid argument type for action %s: expected dict, got %s",
                        _action_id,
                        type(args).__name__
                    )
                    raise GenesysToolError("Tool arguments must be a JSON object")
                
                try:
                    args_json_str = json.dumps(args)
                    payload_bytes = len(args_json_str.encode('utf-8'))
                except (TypeError, ValueError) as json_err:
                    session_logger.error(
                        "[GenesysTools] ERROR: Failed to serialize arguments for action %s: %s",
                        _action_id,
                        json_err,
                        exc_info=True
                    )
                    raise GenesysToolError(f"Failed to serialize tool arguments: {str(json_err)}")
                
                if payload_bytes > GENESYS_MAX_TOOL_ARGUMENT_BYTES:
                    session_logger.error(
                        "[GenesysTools] ERROR: Arguments too large for action %s: %s bytes (max: %s)",
                        _action_id,
                        payload_bytes,
                        GENESYS_MAX_TOOL_ARGUMENT_BYTES
                    )
                    raise GenesysToolError("Tool arguments payload is too large")
                
                invocation_counter['count'] += 1
                session_logger.info(
                    "[GenesysTools] Executing data action %s (call #%s) with %s bytes of arguments",
                    _action_id,
                    invocation_counter['count'],
                    payload_bytes
                )
                
                try:
                    result_payload = await GENESYS_ACTIONS_CLIENT.execute(_action_id, args)
                except GenesysToolError as api_err:
                    session_logger.error(
                        "[GenesysTools] ERROR: API error executing action %s: %s",
                        _action_id,
                        api_err,
                        exc_info=True
                    )
                    raise
                except Exception as exec_err:
                    session_logger.error(
                        "[GenesysTools] ERROR: Unexpected error executing action %s: %s",
                        _action_id,
                        exec_err,
                        exc_info=True
                    )
                    raise GenesysToolError(f"Failed to execute data action: {str(exec_err)}")
                
                if result_payload is None:
                    session_logger.warning(
                        "[GenesysTools] WARNING: Action %s returned None, treating as empty result",
                        _action_id
                    )
                    result_payload = {}
                
                try:
                    redacted = _redact_payload(result_payload)
                except Exception as redact_err:
                    session_logger.error(
                        "[GenesysTools] ERROR: Redaction failed for action %s (%s); withholding the result from the model",
                        _action_id,
                        type(redact_err).__name__
                    )
                    raise GenesysToolError("The data action result could not be redacted and was withheld")
                
                session_logger.info(
                    "[GenesysTools] Successfully executed action %s",
                    _action_id
                )
                
                return {
                    "status": "ok",
                    "action_id": _action_id,
                    "result": redacted,
                    "schema": _success_schema
                }
            except GenesysToolError:
                raise
            except Exception as unexpected_err:
                session_logger.error(
                    "[GenesysTools] CRITICAL ERROR: Unexpected exception in handler for action %s: %s",
                    _action_id,
                    unexpected_err,
                    exc_info=True
                )
                raise GenesysToolError(f"Internal error in tool handler: {str(unexpected_err)}")

        handlers[tool_name] = handler

    if not tools:
        session_logger.warning("[GenesysTools] No valid data action tools could be prepared")
        return None

    instructions = _build_instruction_text(tool_summaries)
    return GenesysToolContext(tools=tools, instructions=instructions, handlers=handlers)
