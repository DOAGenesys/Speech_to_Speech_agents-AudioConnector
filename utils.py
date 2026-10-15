import json
import audioop
import re
from typing import Callable, Optional

try:
    import numpy as np
    import soxr
except ImportError:
    np = None
    soxr = None

from config import (
    MASTER_SYSTEM_PROMPT,
    LANGUAGE_SYSTEM_PROMPT,
    GENESYS_PCMU_FRAME_SIZE,
    GENESYS_PCMU_SILENCE_BYTE,
    logger
)
from call_control import CALL_CONTROL_GUIDANCE

def is_websocket_open(ws) -> bool:
    """
    Backward-compatible check for websocket open state.
    Works with both websockets 15.x (state enum) and older versions (open attribute).

    WEBSOCKETS VERSION COMPATIBILITY:
    - websockets >= 15.0: Removed 'open' boolean attribute, use State enum instead
      * ws.state == State.OPEN checks if connection is open
      * State enum values: CONNECTING(0), OPEN(1), CLOSING(2), CLOSED(3)

    - websockets < 15.0: Used 'open' boolean attribute
      * ws.open returns True if the connection is open, False otherwise

    This function tries the new approach first, then falls back to old approach
    for maximum compatibility across all websocket library versions.

    :param ws: WebSocket connection object
    :return: True if websocket is open, False otherwise
    """
    if ws is None:
        return False

    # Try websockets 15.x+ approach (state enum)
    # The 'open' attribute was removed in v15.0, replaced with state enum
    try:
        from websockets.protocol import State
        if hasattr(ws, 'state'):
            return ws.state == State.OPEN
    except (ImportError, AttributeError):
        pass

    # Fall back to older websockets versions (< 15.0) using 'open' boolean attribute
    if hasattr(ws, 'open'):
        return ws.open

    # If neither method works, assume closed for safety
    return False

def get_websocket_path(ws) -> str:
    """
    Backward-compatible method to extract WebSocket path.
    Works across different websockets library versions and object types.

    WEBSOCKETS VERSION COMPATIBILITY:
    - websockets 12.x: In process_request callback, the first parameter IS the path string
    - websockets >= 13.0: In connection handlers, ws.request.path nested attribute
    - websockets < 12.0: Path may be directly on ws.path

    This function tries multiple approaches to retrieve the path
    for maximum compatibility across all websocket library versions.

    :param ws: Path string (12.x), WebSocket connection object, OR Request object
    :return: Path string if available, otherwise "Not available"
    """
    if ws is None:
        return "Not available"

    # In websockets 12.x, process_request receives the path as a string directly
    # Check if the parameter is already a string (the path itself)
    if isinstance(ws, str):
        return ws

    # Try the direct path attribute first (Request object passed from process_request)
    # This handles the case where ws is actually a Request object, not ServerConnection
    # Request objects have 'path' but no 'state' attribute
    # ServerConnection objects have 'state' but path is nested under 'request'
    if hasattr(ws, 'path') and not hasattr(ws, 'state'):
        path = getattr(ws, 'path', None)
        if path is not None and isinstance(path, str):
            return path

    # Try ws.request.path (websockets 13.x+ for ServerConnection in connection handlers)
    if hasattr(ws, 'request') and hasattr(ws.request, 'path'):
        return ws.request.path

    # Try the direct ws.path attribute (older versions where path was on connection object)
    if hasattr(ws, 'path'):
        path = getattr(ws, 'path', None)
        if path is not None and isinstance(path, str):
            return path

    # Could not determine a path
    return "Not available"

def get_websocket_state_name(ws) -> str:
    """
    Get a human-readable WebSocket state name.
    Works across different websockets library versions.

    WEBSOCKETS VERSION COMPATIBILITY:
    - websockets >= 15.0: Uses State enum (CONNECTING=0, OPEN=1, CLOSING=2, CLOSED=3)
    - websockets < 15.0: May have 'open' boolean or numeric state

    :param ws: WebSocket connection object
    :return: State name string (e.g., "OPEN (1)", "CLOSED (3)", etc.)
    """
    if ws is None:
        return "UNKNOWN"

    # Try to get state value
    state_value = getattr(ws, 'state', None)

    if state_value is not None:
        # Try to import State enum for name lookup
        try:
            from websockets.protocol import State
            state_names = {
                State.CONNECTING: "CONNECTING",
                State.OPEN: "OPEN",
                State.CLOSING: "CLOSING",
                State.CLOSED: "CLOSED"
            }
            # If state_value is an enum member, get its name
            if hasattr(state_value, 'name'):
                return f"{state_value.name} ({state_value.value})"
            # Otherwise treat as integer
            elif isinstance(state_value, int):
                # Map integer to name
                int_to_name = {0: "CONNECTING", 1: "OPEN", 2: "CLOSING", 3: "CLOSED"}
                name = int_to_name.get(state_value, "UNKNOWN")
                return f"{name} ({state_value})"
        except (ImportError, AttributeError):
            pass

        # Fallback: just return the numeric value
        if isinstance(state_value, int):
            int_to_name = {0: "CONNECTING", 1: "OPEN", 2: "CLOSING", 3: "CLOSED"}
            name = int_to_name.get(state_value, "UNKNOWN")
            return f"{name} ({state_value})"

        return str(state_value)

    # Fall back to 'open' attribute for older versions
    if hasattr(ws, 'open'):
        return "OPEN" if ws.open else "CLOSED"

    return "UNKNOWN"

def extract_request_headers(connection, request):
    """
    Extract request headers from process_request callback parameters.
    Works across different websockets library versions.

    WEBSOCKETS VERSION COMPATIBILITY:
    - websockets 12.x: process_request(path: str, request_headers: Headers)
      * First param is the path string
      * Second param is the Headers object directly

    - websockets 13.x-14.x: process_request(connection: ServerConnection, request: Request)
      * First param is ServerConnection object
      * Second param is Request object with request.headers

    - websockets >= 15.0: process_request(connection: ServerConnection)
      * Only one param - ServerConnection object
      * Headers accessed via connection.request.headers

    :param connection: First parameter from process_request callback (path string or ServerConnection)
    :param request: Second parameter from process_request callback (Headers or Request object, or None)
    :return: Headers object or empty dict if not found
    """
    # Case 1: websockets 12.x - connection is actually the path string, request is headers
    if isinstance(connection, str):
        return request if request is not None else {}

    # Case 2: websockets 13.x-14.x - request is a Request object with headers attribute
    if request is not None and hasattr(request, 'headers'):
        return request.headers

    # Case 3: websockets >= 15.0 - only connection param, headers nested in connection.request.headers
    if hasattr(connection, 'request') and hasattr(connection.request, 'headers'):
        return connection.request.headers

    # Fallback: return empty dict if no headers found
    return {}

def get_websocket_connect_kwargs(url, headers, **other_kwargs):
    """
    Build kwargs for websockets.connect() that work with both client implementations.

    - websockets >= 14 (asyncio implementation, the default ``websockets.connect``): ``additional_headers``
    - legacy client (websockets <= 13 default, ``websockets.legacy``): ``extra_headers``

    :param url: WebSocket URL to connect to
    :param headers: Dict of headers to include in the connection
    :param other_kwargs: Additional kwargs to pass to websockets.connect()
    :return: Dict of kwargs ready to pass to websockets.connect()
    """
    import inspect
    import websockets

    kwargs = {"uri": url}
    kwargs.update(other_kwargs)

    try:
        param_names = inspect.signature(websockets.connect).parameters
    except (TypeError, ValueError):
        param_names = {}
    header_param = "extra_headers" if "extra_headers" in param_names and "additional_headers" not in param_names else "additional_headers"
    kwargs[header_param] = headers
    return kwargs

class Pcm16Resampler:
    """
    Streaming mono PCM16 resampler that keeps filter state across chunks.

    Uses soxr (band-limited, no aliasing when downsampling 24 kHz model audio to 8 kHz telephony)
    and falls back to audioop.ratecv when soxr is not installed.
    """

    def __init__(self, in_rate: int, out_rate: int):
        self.in_rate = in_rate
        self.out_rate = out_rate
        self._pending_byte = b""
        self._ratecv_state = None
        self._stream = None
        if in_rate != out_rate and soxr is not None:
            self._stream = soxr.ResampleStream(in_rate, out_rate, 1, dtype="int16")

    def process(self, pcm16: bytes, last: bool = False) -> bytes:
        data = self._pending_byte + (pcm16 or b"")
        usable = len(data) - (len(data) % 2)
        self._pending_byte = data[usable:]
        data = data[:usable]
        if self.in_rate == self.out_rate:
            return data
        if self._stream is not None:
            if not data and not last:
                return b""
            samples = np.frombuffer(data, dtype="<i2")
            return self._stream.resample_chunk(samples, last=last).astype("<i2").tobytes()
        if not data:
            return b""
        out, self._ratecv_state = audioop.ratecv(data, 2, 1, self.in_rate, self.out_rate, self._ratecv_state)
        return out

    def reset(self):
        self._pending_byte = b""
        self._ratecv_state = None
        if self._stream is not None:
            self._stream.clear()


class PcmuFrameAssembler:
    """Re-frames an arbitrary PCMU byte stream into fixed-size Genesys AudioHook frames."""

    def __init__(
        self,
        emit: Callable[[bytes], None],
        frame_size: int = GENESYS_PCMU_FRAME_SIZE,
        silence_byte: int = GENESYS_PCMU_SILENCE_BYTE,
    ):
        self._emit = emit
        self._frame_size = frame_size
        self._silence = bytes([silence_byte])
        self._buffer = bytearray()

    def push(self, pcmu: bytes):
        if pcmu:
            self._buffer.extend(pcmu)
        while len(self._buffer) >= self._frame_size:
            frame = bytes(self._buffer[:self._frame_size])
            del self._buffer[:self._frame_size]
            self._emit(frame)

    def flush(self):
        """Emit any partial frame padded with silence (call at the end of each spoken response)."""
        if not self._buffer:
            return
        self._buffer.extend(self._silence * (self._frame_size - len(self._buffer)))
        frame = bytes(self._buffer)
        self._buffer.clear()
        self._emit(frame)

    def clear(self):
        self._buffer.clear()

    @property
    def pending_bytes(self) -> int:
        return len(self._buffer)


class TokenUsage:
    """Cumulative token counters reported to Genesys as TOTAL_* output variables."""

    FIELDS = (
        "input_text_tokens",
        "input_cached_text_tokens",
        "input_audio_tokens",
        "input_cached_audio_tokens",
        "output_text_tokens",
        "output_audio_tokens",
    )

    def __init__(self):
        self.totals = dict.fromkeys(self.FIELDS, 0)

    def add(self, **counts):
        for name, value in counts.items():
            try:
                self.totals[name] += int(value or 0)
            except (TypeError, ValueError):
                continue

    def as_output_variables(self) -> dict:
        return {f"TOTAL_{name.upper()}": str(value) for name, value in self.totals.items()}


_BCP47_TAG = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")


def as_bcp47_language_tag(value: Optional[str]) -> Optional[str]:
    """Return ``value`` when it looks like a BCP-47 tag (``es``, ``pt-BR``), otherwise None (e.g. ``Spanish``)."""
    if not value:
        return None
    candidate = value.strip()
    return candidate if _BCP47_TAG.match(candidate) else None


def format_json(obj: dict) -> str:
    return json.dumps(obj, indent=2)

def create_final_system_prompt(admin_prompt, language=None, customer_data=None, agent_name=None, company_name=None,
                               call_control_guidance=CALL_CONTROL_GUIDANCE):
    base_prompt = LANGUAGE_SYSTEM_PROMPT.format(language=language) if language else MASTER_SYSTEM_PROMPT
    admin_prompt = admin_prompt or "You are a helpful assistant."

    if agent_name:
        admin_prompt = admin_prompt.replace("[AGENT_NAME]", agent_name)
    if company_name:
        admin_prompt = admin_prompt.replace("[COMPANY_NAME]", company_name)
        admin_prompt = admin_prompt.replace("Our Company", company_name)

    customer_instructions = ""
    if customer_data:
        try:
            data_pairs = [pair.strip() for pair in customer_data.split(';')]
            data_dict = {}
            for pair in data_pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    data_dict[key.strip()] = value.strip()

            if data_dict:
                customer_instructions = "\n\n[CUSTOMER DATA - USE WHEN APPROPRIATE]\n"
                for key, value in data_dict.items():
                    customer_instructions += f"{key}: {value}\n"
                customer_instructions += "Use this customer data to personalize the conversation when relevant."
        except Exception as e:
            logger.warning(f"Error parsing customer data: {type(e).__name__}")

    prompt = f"""[TIER 1 - MASTER INSTRUCTIONS - HIGHEST PRIORITY]
{base_prompt}

[TIER 2 - ADMIN INSTRUCTIONS]
{admin_prompt}{customer_instructions}

[HIERARCHY ENFORCEMENT]
In case of any conflict between Tier 1 and Tier 2 instructions, Tier 1 (Master) instructions 
MUST ALWAYS take precedence and override any conflicting Tier 2 instructions.
"""
    if call_control_guidance:
        prompt += f"\n{call_control_guidance}\n"
    return prompt

def parse_iso8601_duration(duration_str: str) -> float:
    match = re.match(r'P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?', duration_str)
    if not match:
        raise ValueError(f"Invalid ISO 8601 duration format: {duration_str}")
    days, hours, minutes, seconds = match.groups()
    total_seconds = 0
    if days:
        total_seconds += int(days) * 86400
    if hours:
        total_seconds += int(hours) * 3600
    if minutes:
        total_seconds += int(minutes) * 60
    if seconds:
        total_seconds += float(seconds)
    return total_seconds
