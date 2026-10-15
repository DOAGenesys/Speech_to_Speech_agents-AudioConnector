"""
OpenAI GPT-Live client (``gpt-live-1``) for the Genesys AudioHook bridge.

GPT-Live is a full-duplex voice model that decides on its own when to listen, backchannel and speak. Reasoning
and tool use are delegated to a Responses backend model configured at session start; function calls arrive as
nested Responses events inside ``response.event`` envelopes and their results go back with
``response.item.create`` + ``response.create``.
"""

import asyncio
import audioop
import base64
import json
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import websockets

from config import (
    OPENAI_API_BASE_URL,
    OPENAI_LIVE_URL,
    OPENAI_LIVE_DELEGATION_MODEL,
    OPENAI_LIVE_DELEGATION_REASONING_EFFORT,
    OPENAI_LIVE_SERVICE_TIER,
)
from call_control import CallControlResolution, call_control_function_tools, is_call_control
from openai_client import OpenAIRealtimeClient, RealtimeRateLimitedError
from utils import create_final_system_prompt, is_websocket_open
from voice_client_base import PendingFunctionCall

GPT_LIVE_DEFAULT_MODEL = "gpt-live-1"
LIVE_SERVICE_TIERS = ("auto", "default", "flex", "priority")
VOICED_RMS_THRESHOLD = 300
FAREWELL_QUIET_SECONDS = 1.0
AUDIO_IDLE_FLUSH_SECONDS = 0.6
SESSION_CLOSE_TIMEOUT_SECONDS = 5.0
_BACKEND_TERMINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed", "response.done")

BACKEND_VOICE_CONTEXT = """## Voice conversation context
You are helping an assistant in a live phone conversation. Transcripts can contain mistakes, unfinished phrases,
and later corrections. Use the latest context and verified records. If a needed detail is still unclear, ask for
that detail instead of guessing."""

BACKEND_RESULT_GUIDANCE = """## Return the result
Return the relevant facts, the task's current status, and the next step. Report an action as complete after the
tool or service confirms success. If the outcome is unclear, state that and explain what needs to be checked."""


def is_gpt_live_model(model: Optional[str]) -> bool:
    return bool(model) and model.strip().lower().startswith("gpt-live")


def _first_sentence(text: str, limit: int = 160) -> str:
    sentence = (text or "").strip().split(". ")[0].strip()
    return sentence[:limit].rstrip(".")


class GptLiveClient(OpenAIRealtimeClient):
    vendor = "openai"
    vendor_label = "GPT-Live"
    default_model = GPT_LIVE_DEFAULT_MODEL

    def __init__(self, session_id: str, on_speech_started_callback=None, api_key: Optional[str] = None):
        super().__init__(session_id, on_speech_started_callback=on_speech_started_callback, api_key=api_key)
        self.delegation_model = OPENAI_LIVE_DELEGATION_MODEL
        self.live_session_id: Optional[str] = None
        self.voice_seconds = 0.0
        self._session_started = False
        self._session_close_sent = False
        self._session_closed = asyncio.Event()
        self._event_counter = 0
        self._backend_calls: Dict[str, Dict[str, PendingFunctionCall]] = {}
        self._transcript: List[Tuple[str, str]] = []
        self._last_audio_delta_at: Optional[float] = None
        self._last_voiced_audio_at: Optional[float] = None
        self._farewell_sent_at: Optional[float] = None
        self._idle_flush_task: Optional[asyncio.Task] = None

    def _build_url(self) -> str:
        return OPENAI_LIVE_URL

    def _next_event_id(self, prefix: str) -> str:
        self._event_counter += 1
        return f"{prefix}_{self._event_counter}"

    # ------------------------------------------------------------------ session setup

    async def connect(
        self,
        instructions=None,
        voice=None,
        temperature=None,
        model=None,
        max_output_tokens=None,
        agent_name=None,
        company_name=None,
        tool_definitions=None,
        tool_instructions=None,
        tool_choice=None,
    ):
        if not self.api_key:
            raise RuntimeError(f"{self.api_key_env_name} is not configured")

        self.model = (model or self.default_model).strip()
        self.voice = (voice or "").strip() or self.default_voice
        self.max_output_tokens = self._parse_max_output_tokens(max_output_tokens)
        self.tool_choice = tool_choice or "auto"

        tools = self._build_tools(tool_definitions)
        live_instructions = self._build_live_instructions(instructions, agent_name, company_name, tools)
        backend_instructions = self._build_backend_instructions(instructions, agent_name, company_name, tool_instructions)
        session = {
            "model": self.model,
            "instructions": live_instructions,
            "audio": {
                "format": {"type": "audio/pcmu", "rate": 8000},
                "output": {"voice": self.voice},
            },
            "delegation": {"type": "responses", "responses": self._build_delegation_config(backend_instructions, tools)},
        }

        try:
            await self._open_websocket()
            await self._send_event({"type": "session.start", "event_id": "session_start", "session": session})
            started = await self._await_setup_event({"session.started"})
        except RealtimeRateLimitedError as exc:
            await self.close()
            raise RuntimeError("GPT-Live rate limit reached while starting the session") from exc
        except Exception as exc:
            await self.close()
            raise RuntimeError(f"Failed to connect to GPT-Live: {type(exc).__name__}: {exc}") from exc

        self._session_started = True
        self.live_session_id = (started.get("session") or {}).get("id")
        self.logger.info(
            f"GPT-Live session ready: id={self.live_session_id} model={self.model} voice={self.voice} "
            f"backend={self.delegation_model} tools={len(tools)}"
        )

    def _build_tools(self, custom_tools: Optional[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        tools = []
        for tool in call_control_function_tools():
            tools.append({**tool, "strict": True})
        for tool in custom_tools or []:
            converted = dict(tool)
            if converted.get("type") == "function":
                converted.setdefault("strict", False)
            elif converted.get("type") == "mcp" and converted.get("require_approval") not in ("never", None):
                self.logger.warning(
                    f"MCP server {converted.get('server_label')} requires approval; GPT-Live approvals are rejected "
                    "automatically, set require_approval to 'never' for phone calls"
                )
            tools.append(converted)
        return tools

    def _build_delegation_config(self, backend_instructions: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "model": self.delegation_model,
            "instructions": backend_instructions,
            "tools": tools,
            "tool_choice": self.tool_choice,
            "parallel_tool_calls": True,
        }
        effort = (OPENAI_LIVE_DELEGATION_REASONING_EFFORT or "").strip().lower()
        if effort:
            config["reasoning"] = {"effort": effort}
        tier = (OPENAI_LIVE_SERVICE_TIER or "").strip().lower()
        if tier in LIVE_SERVICE_TIERS:
            config["service_tier"] = tier
        if self.max_output_tokens is not None and self.max_output_tokens >= 16:
            config["max_output_tokens"] = self.max_output_tokens
        return config

    def _build_live_instructions(self, instructions, agent_name, company_name, tools) -> str:
        persona = create_final_system_prompt(
            instructions,
            language=self.language,
            customer_data=self.customer_data,
            agent_name=agent_name,
            company_name=company_name,
            call_control_guidance=None,
        )
        capabilities = [
            "- Call control: end the call once the caller's request is resolved and they confirm they need nothing "
            "else, or transfer the caller to a human agent."
        ]
        for tool in tools:
            if tool.get("type") == "function" and not is_call_control(tool.get("name")):
                capabilities.append(f"- {tool['name']}: {_first_sentence(tool.get('description', '')) or 'backend action'}.")
            elif tool.get("type") == "mcp":
                capabilities.append(f"- {tool.get('server_label', 'MCP server')}: tools from a connected service.")
            elif tool.get("type") in ("web_search", "file_search"):
                capabilities.append(f"- {tool['type'].replace('_', ' ').capitalize()}: look up information.")

        return "\n\n".join([
            persona.strip(),
            "Backchannel policy: Use few backchannels. Acknowledge briefly without competing with the caller.",
            "Interruption policy: Stop speaking when the caller interrupts. Listen to what they say.",
            "\n".join([
                "Delegation policy:",
                "Backend tools:",
                *capabilities,
                "",
                "Delegate to the backend when:",
                "- The caller confirms they are done or asks to end the call.",
                "- The caller asks for a human agent, is clearly dissatisfied, or the request cannot be completed.",
                "- The request needs a backend capability listed above or careful reasoning.",
                "- A correction changes work already requested.",
                "",
                "Do not delegate to the backend when:",
                "- You can answer from the conversation or a still-current result.",
                "- You need a brief clarification to understand the request.",
                "",
                "Delegate before giving an answer that depends on backend work.",
                "Do not guess the result while waiting.",
                "Do not say goodbye on your own; a closing instruction arrives when the call is ending.",
            ]),
        ])

    def _build_backend_instructions(self, instructions, agent_name, company_name, tool_instructions) -> str:
        task = create_final_system_prompt(
            instructions,
            language=self.language,
            customer_data=self.customer_data,
            agent_name=agent_name,
            company_name=company_name,
        )
        sections = [BACKEND_VOICE_CONTEXT, f"## Task instructions\n{task.strip()}"]
        if tool_instructions:
            sections.append(tool_instructions.strip())
        sections.append(BACKEND_RESULT_GUIDANCE)
        return "\n\n".join(sections)

    # ------------------------------------------------------------------ audio

    async def send_audio(self, pcmu_8k: bytes):
        if not self.running or not self._session_started or self._session_close_sent or not pcmu_8k:
            return
        try:
            await self._send_event({
                "type": "session.input_audio.append",
                "audio": base64.b64encode(pcmu_8k).decode("ascii"),
            })
        except websockets.exceptions.ConnectionClosed:
            self.running = False

    async def start_receiving(self, on_audio_callback):
        await super().start_receiving(on_audio_callback)
        self._idle_flush_task = self._spawn(self._flush_idle_audio())

    async def _flush_idle_audio(self):
        while self.running:
            await asyncio.sleep(0.1)
            last = self._last_audio_delta_at
            if (self._audio_assembler and self._audio_assembler.pending_bytes and last is not None
                    and time.monotonic() - last >= AUDIO_IDLE_FLUSH_SECONDS):
                self._audio_assembler.flush()

    def _handle_live_audio_delta(self, event: Dict[str, Any]):
        data = event.get("delta")
        if not data or self._audio_assembler is None:
            return
        pcmu = base64.b64decode(data)
        now = time.monotonic()
        self._last_audio_delta_at = now
        if audioop.rms(audioop.ulaw2lin(pcmu, 2), 2) >= VOICED_RMS_THRESHOLD:
            self._last_voiced_audio_at = now
        self._audio_assembler.push(pcmu)

    # ------------------------------------------------------------------ events

    async def _dispatch(self, event: Dict[str, Any]):
        event_type = event.get("type", "")

        if event_type == "session.output_audio.delta":
            self._handle_live_audio_delta(event)
        elif event_type == "session.input_transcript.delta":
            self._append_transcript("Caller", event.get("delta"))
        elif event_type == "session.output_transcript.delta":
            self._append_transcript("Agent", event.get("delta"))
            if self._farewell_sent_at is not None:
                self._last_voiced_audio_at = time.monotonic()
        elif event_type == "response.event":
            await self._handle_backend_event(event.get("delegation_id"), event.get("event") or {})
        elif event_type == "session.delegation.created":
            delegation = event.get("delegation") or {}
            self.logger.info(f"Delegation {delegation.get('id')} started (target={delegation.get('target')})")
        elif event_type == "session.usage.updated":
            seconds = (event.get("usage") or {}).get("seconds")
            if isinstance(seconds, (int, float)):
                self.voice_seconds = float(seconds)
        elif event_type == "session.closed":
            seconds = (event.get("usage") or {}).get("seconds")
            if isinstance(seconds, (int, float)):
                self.voice_seconds = float(seconds)
            self.logger.info(f"GPT-Live session closed (reason={event.get('reason')}, voice_seconds={self.voice_seconds})")
            self._session_closed.set()
        elif event_type == "error":
            self._handle_live_error(event)
        elif event_type in ("session.instructions.appended", "session.updated", "session.started"):
            self.logger.debug(event_type)

    def _append_transcript(self, speaker: str, delta: Optional[str]):
        if not delta:
            return
        if self._transcript and self._transcript[-1][0] == speaker:
            self._transcript[-1] = (speaker, self._transcript[-1][1] + delta)
        else:
            self._transcript.append((speaker, delta))

    def _handle_live_error(self, event: Dict[str, Any]):
        error = event.get("error") or {}
        code = error.get("code")
        message = error.get("message")
        if code == "responses_validation_failed":
            self.logger.error(f"GPT-Live rejected the backend configuration: {message}")
            return
        self.logger.warning(
            f"GPT-Live error: type={error.get('type')} code={code} param={error.get('param')} "
            f"client_event_id={error.get('client_event_id')} message={message}"
        )

    async def _handle_backend_event(self, delegation_id: Optional[str], inner: Dict[str, Any]):
        inner_type = inner.get("type", "")
        response = inner.get("response") or {}
        # Nested output-item events carry no response id; the envelope's delegation id ties them together.
        delegation_key = delegation_id or ""

        if inner_type == "response.output_item.done":
            item = inner.get("item") or {}
            item_type = item.get("type")
            if item_type == "function_call" and item.get("call_id") and item.get("name"):
                bucket = self._backend_calls.setdefault(delegation_key, {})
                bucket.setdefault(item["call_id"], PendingFunctionCall(item["call_id"], item["name"], item.get("arguments")))
            elif item_type == "mcp_approval_request":
                await self._reject_mcp_approval(item)
            elif item_type == "mcp_call":
                self.logger.info(f"[MCP] {item.get('server_label')}.{item.get('name')} finished")
            return

        if inner_type in _BACKEND_TERMINAL_EVENTS:
            self._accumulate_backend_usage(response.get("usage") or {})
            if inner_type == "response.failed":
                error = response.get("error") or {}
                self.logger.warning(f"Backend response failed: code={error.get('code')} message={error.get('message')}")
            calls = [
                call for call_id, call in self._backend_calls.pop(delegation_key, {}).items()
                if call_id not in self._handled_call_ids
            ]
            self._handled_call_ids.update(call.call_id for call in calls)
            if calls:
                self._spawn(self._process_function_calls(calls))

    async def _reject_mcp_approval(self, item: Dict[str, Any]):
        self.logger.warning(f"Rejecting MCP approval request for {item.get('server_label')}.{item.get('name')}")
        await self._send_event({
            "type": "response.item.create",
            "event_id": self._next_event_id("mcp_approval"),
            "item": {"type": "mcp_approval_response", "approval_request_id": item.get("id"), "approve": False},
        })
        await self._send_event({"type": "response.create", "event_id": self._next_event_id("continue")})

    def _accumulate_backend_usage(self, usage: Dict[str, Any]):
        if not usage:
            return
        cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
        input_tokens = usage.get("input_tokens") or 0
        self.usage.add(
            input_text_tokens=max(0, int(input_tokens) - int(cached)),
            input_cached_text_tokens=cached,
            output_text_tokens=usage.get("output_tokens"),
        )

    # ------------------------------------------------------------------ function calling

    async def _send_function_output(self, call_id: str, payload: Dict[str, Any]):
        await self._send_event({
            "type": "response.item.create",
            "event_id": self._next_event_id("tool_result"),
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(payload, ensure_ascii=False, default=str),
            },
        })

    async def _continue_after_tool_outputs(self):
        await self._send_event({"type": "response.create", "event_id": self._next_event_id("continue")})

    async def _send_farewell(self, resolution: CallControlResolution):
        self._farewell_sent_at = time.monotonic()
        await self._send_event({
            "type": "session.instructions.append",
            "event_id": self._next_event_id("farewell"),
            "delegation_id": None,
            "content": f"The call is ending now. {resolution.farewell_instruction} Do not ask any further questions.",
        })
        self._spawn(self._await_farewell_spoken())

    async def _await_farewell_spoken(self):
        while not self._disconnect_triggered and self.running:
            await asyncio.sleep(0.1)
            voiced_at = self._last_voiced_audio_at
            if (voiced_at is not None and self._farewell_sent_at is not None and voiced_at > self._farewell_sent_at
                    and time.monotonic() - voiced_at >= FAREWELL_QUIET_SECONDS):
                await self._finish_farewell("farewell spoken")
                return

    # ------------------------------------------------------------------ summary, metrics, close

    async def generate_summary(self, prompt: str, timeout: float = 10.0) -> Optional[str]:
        transcript = "\n".join(f"{speaker}: {text.strip()}" for speaker, text in self._transcript if text.strip())
        if not transcript:
            return None
        body: Dict[str, Any] = {
            "model": self.delegation_model,
            "instructions": prompt,
            "input": f"Conversation transcript:\n{transcript}",
            "max_output_tokens": 600,
            "store": False,
        }
        effort = (OPENAI_LIVE_DELEGATION_REASONING_EFFORT or "").strip().lower()
        if effort:
            body["reasoning"] = {"effort": effort}
        try:
            async with httpx.AsyncClient(timeout=timeout) as http:
                response = await http.post(
                    f"{OPENAI_API_BASE_URL.rstrip('/')}/responses",
                    headers=self._auth_headers(),
                    json=body,
                )
        except httpx.HTTPError as exc:
            self.logger.warning(f"Summary request failed: {type(exc).__name__}")
            return None
        if response.status_code >= 400:
            self.logger.warning(f"Summary request failed with HTTP {response.status_code}")
            return None
        try:
            data = response.json()
        except ValueError:
            return None
        self._accumulate_backend_usage(data.get("usage") or {})
        text = data.get("output_text")
        if not text:
            text = "".join(
                part.get("text", "")
                for item in data.get("output") or [] if item.get("type") == "message"
                for part in item.get("content") or [] if part.get("type") == "output_text"
            )
        return text.strip() or None

    def get_token_metrics(self) -> Dict[str, str]:
        metrics = super().get_token_metrics()
        metrics["TOTAL_VOICE_SECONDS"] = f"{self.voice_seconds:g}"
        return metrics

    async def close(self):
        if (self._session_started and not self._session_close_sent and self.ws is not None
                and is_websocket_open(self.ws) and self.read_task is not None and not self.read_task.done()):
            self._session_close_sent = True
            self._closing = True
            try:
                await self._send_event({"type": "session.close", "event_id": "session_close"})
                await asyncio.wait_for(self._session_closed.wait(), timeout=SESSION_CLOSE_TIMEOUT_SECONDS)
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                self.logger.warning("session.closed not received; final GPT-Live voice usage is unconfirmed")
        await super().close()
