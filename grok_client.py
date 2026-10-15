"""
xAI Grok Voice Agent API client (``grok-voice-3.0``) for the Genesys AudioHook bridge.

The Voice Agent API is OpenAI Realtime-compatible, so this client reuses ``OpenAIRealtimeClient`` and only
overrides the session schema, tool format, farewell delivery and summary request.
"""

import copy
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote

from config import (
    XAI_API_KEY,
    GROK_MODEL,
    GROK_VOICE,
    XAI_REALTIME_BASE_URL,
    GROK_REASONING_EFFORT,
    GROK_IDLE_TIMEOUT_MS,
    GROK_SPEECH_SPEED,
    GROK_KEYTERMS,
)
from call_control import CallControlResolution, call_control_function_tools
from openai_client import OpenAIRealtimeClient
from utils import as_bcp47_language_tag

GROK_BUILTIN_TOOL_KEYS = {
    "web_search": {"type", "location", "user_location", "allowed_domains", "excluded_domains", "enable_image_understanding"},
    "x_search": {"type", "allowed_x_handles", "excluded_x_handles", "from_date", "to_date",
                 "enable_image_understanding", "enable_video_understanding"},
    "file_search": {"type", "vector_store_ids", "max_num_results"},
    "mcp": {"type", "server_label", "server_url", "allowed_tools", "authorization", "headers"},
}
_TOOL_TYPE_ALIASES = {"web_search_preview": "web_search"}
MAX_KEYTERMS = 100
MAX_KEYTERM_LENGTH = 50


def _parse_keyterms(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    terms = [term.strip() for term in str(raw).replace("\n", ",").split(",")]
    return [term for term in terms if term and len(term) <= MAX_KEYTERM_LENGTH][:MAX_KEYTERMS]


class GrokRealtimeClient(OpenAIRealtimeClient):
    vendor = "grok"
    vendor_label = "Grok"
    api_key_env_name = "XAI_API_KEY"
    default_model = GROK_MODEL
    default_voice = GROK_VOICE
    wait_for_session_created = False
    match_untagged_responses = True

    def __init__(self, session_id: str, on_speech_started_callback=None, api_key: Optional[str] = None):
        super().__init__(session_id, on_speech_started_callback=on_speech_started_callback, api_key=api_key)
        self.keyterms: List[str] = _parse_keyterms(GROK_KEYTERMS)
        self.reasoning_effort: Optional[str] = GROK_REASONING_EFFORT

    def _default_api_key(self) -> Optional[str]:
        return XAI_API_KEY

    def _build_url(self) -> str:
        return f"{XAI_REALTIME_BASE_URL}?model={quote(self.model, safe='')}"

    def set_keyterms(self, raw: Optional[str]):
        """Override the deployment keyterms for this call (e.g. from the GROK_KEYTERMS input variable)."""
        if raw:
            self.keyterms = _parse_keyterms(raw)

    def _build_tools(self, custom_tools: Optional[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        tools = call_control_function_tools()
        for tool in custom_tools or []:
            converted = self._convert_tool(tool)
            if converted is not None:
                tools.append(converted)
        return tools

    def _convert_tool(self, tool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        tool_type = _TOOL_TYPE_ALIASES.get(tool.get("type"), tool.get("type"))
        if tool_type == "function":
            source = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            if not source.get("name"):
                return None
            converted = {"type": "function", "name": source["name"], "description": source.get("description", "")}
            parameters = copy.deepcopy(source.get("parameters") or {"type": "object", "properties": {}})
            parameters.pop("strict", None)
            converted["parameters"] = parameters
            return converted
        allowed_keys = GROK_BUILTIN_TOOL_KEYS.get(tool_type)
        if allowed_keys is None:
            self.logger.warning(f"Skipping tool type '{tool_type}': not supported by the xAI Voice Agent API")
            return None
        converted = {key: copy.deepcopy(value) for key, value in tool.items() if key in allowed_keys}
        converted["type"] = tool_type
        dropped = sorted(set(tool) - allowed_keys)
        if dropped:
            self.logger.info(f"Dropped unsupported {tool_type} tool fields for Grok: {', '.join(dropped)}")
        return converted

    def _build_session_config(self, instructions: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        turn_detection: Dict[str, Any] = {"type": "server_vad"}
        try:
            idle_timeout_ms = int(GROK_IDLE_TIMEOUT_MS) if GROK_IDLE_TIMEOUT_MS else 0
        except ValueError:
            idle_timeout_ms = 0
        if idle_timeout_ms > 0:
            turn_detection["idle_timeout_ms"] = idle_timeout_ms

        audio_input: Dict[str, Any] = {"format": {"type": "audio/pcmu"}}
        transcription: Dict[str, Any] = {}
        language_hint = as_bcp47_language_tag(self.language)
        if language_hint:
            transcription["language_hint"] = language_hint
        if self.keyterms:
            transcription["keyterms"] = self.keyterms
        if transcription:
            audio_input["transcription"] = transcription

        audio_output: Dict[str, Any] = {"format": {"type": "audio/pcmu"}}
        speed = self._parse_speed(GROK_SPEECH_SPEED)
        if speed is not None:
            audio_output["speed"] = speed

        session: Dict[str, Any] = {
            "instructions": instructions,
            "voice": self.voice,
            "turn_detection": turn_detection,
            "audio": {"input": audio_input, "output": audio_output},
            "tools": tools,
        }
        effort = (self.reasoning_effort or "").strip().lower()
        if effort in ("high", "none"):
            session["reasoning"] = {"effort": effort}
        return session

    @staticmethod
    def _parse_speed(raw: Optional[str]) -> Optional[float]:
        if not raw:
            return None
        try:
            speed = float(raw)
        except ValueError:
            return None
        return min(max(speed, 0.7), 1.5)

    async def _send_farewell(self, resolution: CallControlResolution):
        if resolution.exact_farewell:
            await self._send_event({
                "type": "conversation.item.create",
                "item": {
                    "type": "force_message",
                    "role": "assistant",
                    "interruptible": False,
                    "content": [{"type": "output_text", "text": resolution.exact_farewell}],
                },
            })
            return
        await self._send_event({
            "type": "response.create",
            "response": {
                "instructions": resolution.farewell_instruction,
                "metadata": {"type": "final_farewell"},
            },
        })

    async def _send_summary_request(self, prompt: str):
        await self._send_event({
            "type": "response.create",
            "response": {
                "modalities": ["text"],
                "instructions": prompt,
                "metadata": {"type": "ending_analysis"},
            },
        })
