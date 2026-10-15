"""
OpenAI Realtime API client (GA interface, ``gpt-realtime-2.1`` family) for the Genesys AudioHook bridge.

Audio flows natively as G.711 u-law (``audio/pcmu``) in both directions, so no transcoding is needed.
``GrokRealtimeClient`` (grok_client.py) reuses this class because the xAI Voice Agent API speaks the
same protocol with a few differences in the session schema.
"""

import asyncio
import base64
import json
import re
import time
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote

import websockets

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_VOICE,
    OPENAI_REALTIME_BASE_URL,
    OPENAI_REASONING_EFFORT,
    OPENAI_TURN_DETECTION,
    OPENAI_VAD_EAGERNESS,
    OPENAI_NOISE_REDUCTION,
    RATE_LIMIT_MAX_RETRIES,
    RATE_LIMIT_BASE_DELAY,
)
from call_control import (
    CallControlResolution,
    call_control_function_tools,
    is_call_control,
    resolve_call_control,
)
from utils import (
    PcmuFrameAssembler,
    create_final_system_prompt,
    get_websocket_connect_kwargs,
    is_websocket_open,
)
from voice_client_base import BaseVoiceClient, PendingFunctionCall

PCMU_BYTES_PER_MS = 8
SETUP_EVENT_TIMEOUT_SECONDS = 10.0

OPENAI_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")
_REASONING_REALTIME_MODEL = re.compile(r"^gpt-realtime-2(?:$|[.-])")
_IGNORED_ERROR_CODES = {
    "conversation_already_has_active_response",
    "response_cancel_not_active",
    "input_audio_buffer_commit_empty",
}


class RealtimeRateLimitedError(RuntimeError):
    pass


def _handshake_status_code(exc: BaseException) -> Optional[int]:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) or getattr(exc, "status_code", None)


class OpenAIRealtimeClient(BaseVoiceClient):
    vendor = "openai"
    vendor_label = "OpenAI"
    api_key_env_name = "OPENAI_API_KEY"
    default_model = OPENAI_MODEL
    default_voice = OPENAI_VOICE
    wait_for_session_created = True
    # True when response.created does not echo request metadata, so the next response is assumed to be ours.
    match_untagged_responses = False

    def __init__(self, session_id: str, on_speech_started_callback=None, api_key: Optional[str] = None):
        super().__init__(session_id, on_speech_started_callback=on_speech_started_callback, api_key=api_key)
        self.ws = None
        self.read_task: Optional[asyncio.Task] = None
        self._send_lock = asyncio.Lock()
        self.tool_choice = "auto"
        self.max_output_tokens = None

        self._on_audio_callback = None
        self._audio_assembler: Optional[PcmuFrameAssembler] = None
        self._audio_item_id: Optional[str] = None
        self._audio_item_ms = 0.0
        self._audio_item_started_at: Optional[float] = None

        self._response_metadata: Dict[str, Dict[str, Any]] = {}
        self._pending_calls: Dict[str, Dict[str, PendingFunctionCall]] = {}
        self._handled_call_ids = set()

        self._summary_awaiting_response = False
        self._summary_response_id: Optional[str] = None
        self._summary_text: List[str] = []

        self._farewell_awaiting_response = False
        self._farewell_response_id: Optional[str] = None

    # ------------------------------------------------------------------ vendor hooks

    def _default_api_key(self) -> Optional[str]:
        return OPENAI_API_KEY

    def _build_url(self) -> str:
        return f"{OPENAI_REALTIME_BASE_URL}?model={quote(self.model, safe='')}"

    def _auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _is_reasoning_model(self) -> bool:
        # Dated snapshots of the original gpt-realtime (gpt-realtime-2025-08-28) are not reasoning models.
        return bool(_REASONING_REALTIME_MODEL.match(self.model))

    def _turn_detection_config(self) -> Dict[str, Any]:
        mode = (OPENAI_TURN_DETECTION or "semantic_vad").strip().lower()
        if mode == "server_vad":
            return {"type": "server_vad", "create_response": True, "interrupt_response": True}
        config = {"type": "semantic_vad", "create_response": True, "interrupt_response": True}
        eagerness = (OPENAI_VAD_EAGERNESS or "auto").strip().lower()
        if eagerness in ("low", "medium", "high", "auto"):
            config["eagerness"] = eagerness
        return config

    def _build_tools(self, custom_tools: Optional[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return call_control_function_tools() + [dict(tool) for tool in (custom_tools or [])]

    def _build_session_config(self, instructions: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        audio_input: Dict[str, Any] = {
            "format": {"type": "audio/pcmu"},
            "turn_detection": self._turn_detection_config(),
        }
        noise_reduction = (OPENAI_NOISE_REDUCTION or "").strip().lower()
        if noise_reduction in ("near_field", "far_field"):
            audio_input["noise_reduction"] = {"type": noise_reduction}

        session: Dict[str, Any] = {
            "type": "realtime",
            "model": self.model,
            "instructions": instructions,
            "output_modalities": ["audio"],
            "audio": {
                "input": audio_input,
                "output": {"format": {"type": "audio/pcmu"}, "voice": self.voice},
            },
            "tools": tools,
            "tool_choice": self.tool_choice,
        }
        if self.max_output_tokens is not None:
            session["max_output_tokens"] = self.max_output_tokens
        if self._is_reasoning_model():
            effort = (OPENAI_REASONING_EFFORT or "").strip().lower()
            if effort in OPENAI_REASONING_EFFORTS:
                session["reasoning"] = {"effort": effort}
        return session

    async def _send_farewell(self, resolution: CallControlResolution):
        await self._send_event({
            "type": "response.create",
            "response": {
                "instructions": resolution.farewell_instruction,
                "tool_choice": "none",
                "metadata": {"type": "final_farewell"},
            },
        })

    async def _send_summary_request(self, prompt: str):
        await self._send_event({
            "type": "response.create",
            "response": {
                "conversation": "none",
                "output_modalities": ["text"],
                "instructions": prompt,
                "max_output_tokens": 600,
                "metadata": {"type": "ending_analysis"},
            },
        })

    # ------------------------------------------------------------------ connection

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
        if temperature is not None:
            self.logger.debug(f"{self.vendor_label} realtime sessions do not accept temperature; ignoring it")

        prompt = self._build_instructions(instructions, agent_name, company_name, tool_instructions)
        tools = self._build_tools(tool_definitions)
        session = self._build_session_config(prompt, tools)

        attempt = 0
        while True:
            attempt += 1
            try:
                await self._open_websocket()
                if self.wait_for_session_created:
                    await self._await_setup_event({"session.created"})
                await self._send_event({"type": "session.update", "session": session})
                await self._await_setup_event({"session.updated"})
                self.logger.info(
                    f"{self.vendor_label} session ready: model={self.model} voice={self.voice} tools={len(tools)}"
                )
                return
            except RealtimeRateLimitedError:
                await self.close()
                self._closing = False
                if attempt > RATE_LIMIT_MAX_RETRIES:
                    raise RuntimeError(f"{self.vendor_label} rate limit persisted after {RATE_LIMIT_MAX_RETRIES} retries")
                delay = RATE_LIMIT_BASE_DELAY * (2 ** (attempt - 1))
                self.logger.warning(f"{self.vendor_label} rate limited during connect; retrying in {delay}s")
                await asyncio.sleep(delay)
            except Exception as exc:
                await self.close()
                raise RuntimeError(f"Failed to connect to {self.vendor_label}: {type(exc).__name__}: {exc}") from exc

    def _build_instructions(self, instructions, agent_name, company_name, tool_instructions) -> str:
        prompt = create_final_system_prompt(
            instructions,
            language=self.language,
            customer_data=self.customer_data,
            agent_name=agent_name,
            company_name=company_name,
        )
        if tool_instructions:
            prompt = f"{prompt}\n{tool_instructions}"
        return prompt

    @staticmethod
    def _parse_max_output_tokens(value) -> Optional[int]:
        if value is None or str(value).strip().lower() in ("", "inf"):
            return None
        try:
            tokens = int(value)
        except (TypeError, ValueError):
            return None
        return tokens if tokens > 0 else None

    async def _open_websocket(self):
        url = self._build_url()
        connect_kwargs = get_websocket_connect_kwargs(
            url,
            self._auth_headers(),
            max_size=2 ** 24,
            compression=None,
        )
        try:
            self.ws = await asyncio.wait_for(websockets.connect(**connect_kwargs), timeout=10)
        except Exception as exc:
            if _handshake_status_code(exc) == 429:
                raise RealtimeRateLimitedError("rate limited") from exc
            raise
        self.running = True
        self._closing = False

    async def _await_setup_event(self, expected_types: set) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + SETUP_EVENT_TIMEOUT_SECONDS
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"no {'/'.join(sorted(expected_types))} event from {self.vendor_label}")
            raw = await asyncio.wait_for(self.ws.recv(), timeout=remaining)
            if isinstance(raw, bytes):
                continue
            event = json.loads(raw)
            event_type = event.get("type")
            if event_type in expected_types:
                return event
            if event_type == "error":
                error = event.get("error") or {}
                code = error.get("code") or event.get("code")
                message = error.get("message") or event.get("message") or "unknown error"
                if code in (429, "rate_limit_exceeded"):
                    raise RealtimeRateLimitedError(message)
                raise RuntimeError(f"{self.vendor_label} rejected the session ({code}): {message}")
            self.logger.debug(f"Ignoring {event_type} during session setup")

    async def _send_event(self, event: Dict[str, Any]) -> bool:
        if not self.ws or not is_websocket_open(self.ws):
            self.logger.debug(f"Dropping {event.get('type')}: websocket is not open")
            return False
        async with self._send_lock:
            await self.ws.send(json.dumps(event))
        return True

    async def send_audio(self, pcmu_8k: bytes):
        if not self.running or not pcmu_8k:
            return
        try:
            await self._send_event({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcmu_8k).decode("ascii"),
            })
        except websockets.exceptions.ConnectionClosed:
            self.running = False

    async def close(self):
        self._closing = True
        self.running = False
        self._cancel_farewell_timeout()
        self._cancel_background_tasks()
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None
        if self.read_task and self.read_task is not asyncio.current_task() and not self.read_task.done():
            self.read_task.cancel()
            try:
                await self.read_task
            except (asyncio.CancelledError, Exception):
                pass
        self._resolve_summary(None)

    # ------------------------------------------------------------------ receiving

    async def start_receiving(self, on_audio_callback):
        self._on_audio_callback = on_audio_callback
        self._audio_assembler = PcmuFrameAssembler(self._emit_audio_frame)
        self.read_task = asyncio.create_task(self._read_loop())

    def _emit_audio_frame(self, frame: bytes):
        if self._on_audio_callback:
            self._on_audio_callback(frame)

    async def _read_loop(self):
        try:
            async for raw in self.ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    self.logger.warning(f"Ignoring non-JSON message from {self.vendor_label}")
                    continue
                try:
                    await self._dispatch(event)
                except Exception as exc:
                    self.logger.error(
                        f"Error handling {event.get('type')}: {type(exc).__name__}: {exc}", exc_info=True
                    )
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"{self.vendor_label} websocket closed")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.error(f"{self.vendor_label} read loop failed: {type(exc).__name__}: {exc}", exc_info=True)
        finally:
            self.running = False
            self._resolve_summary(None)
            self._notify_connection_lost()

    async def _dispatch(self, event: Dict[str, Any]):
        event_type = event.get("type", "")

        if event_type in ("response.output_audio.delta", "response.audio.delta"):
            self._handle_audio_delta(event)
        elif event_type in ("response.output_audio.done", "response.audio.done"):
            if self._audio_assembler:
                self._audio_assembler.flush()
        elif event_type in ("response.output_text.delta", "response.text.delta",
                            "response.output_audio_transcript.delta", "response.audio_transcript.delta"):
            self._handle_text_delta(event)
        elif event_type == "input_audio_buffer.speech_started":
            await self._handle_speech_started()
        elif event_type == "response.created":
            self._handle_response_created(event.get("response") or {})
        elif event_type == "response.function_call_arguments.done":
            self._record_function_call(
                event.get("response_id"), event.get("call_id"), event.get("name"), event.get("arguments")
            )
        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                self._record_function_call(
                    event.get("response_id"), item.get("call_id"), item.get("name"), item.get("arguments")
                )
        elif event_type == "response.done":
            await self._handle_response_done(event.get("response") or {})
        elif event_type == "error":
            self._handle_error_event(event)
        elif event_type.startswith("response.mcp_call") or event_type.startswith("mcp_list_tools"):
            self._log_mcp_event(event_type, event)
        elif event_type == "input_audio_buffer.timeout_triggered":
            self.logger.info("Caller idle timeout reached; the model will re-engage the caller")
        elif event_type in ("session.created", "session.updated"):
            self.logger.debug(event_type)

    def _handle_audio_delta(self, event: Dict[str, Any]):
        data = event.get("delta")
        if not data or self._audio_assembler is None:
            return
        if self._summary_response_id and event.get("response_id") == self._summary_response_id:
            return
        pcmu = base64.b64decode(data)
        item_id = event.get("item_id")
        if item_id != self._audio_item_id:
            self._audio_item_id = item_id
            self._audio_item_ms = 0.0
            self._audio_item_started_at = time.monotonic()
        self._audio_item_ms += len(pcmu) / PCMU_BYTES_PER_MS
        self._audio_assembler.push(pcmu)

    def _handle_text_delta(self, event: Dict[str, Any]):
        response_id = event.get("response_id") or self._summary_response_id
        if self._summary_response_id and response_id == self._summary_response_id and event.get("delta"):
            self._summary_text.append(event["delta"])

    async def _handle_speech_started(self):
        if self._audio_assembler:
            self._audio_assembler.clear()
        if self.on_speech_started_callback:
            try:
                await self.on_speech_started_callback()
            except Exception as exc:
                self.logger.error(f"Barge-in callback failed: {type(exc).__name__}: {exc}")
        await self._truncate_interrupted_audio()

    async def _truncate_interrupted_audio(self):
        item_id, started_at, total_ms = self._audio_item_id, self._audio_item_started_at, self._audio_item_ms
        self._audio_item_id = None
        self._audio_item_started_at = None
        self._audio_item_ms = 0.0
        if not item_id or started_at is None or self._farewell is not None:
            return
        played_ms = (time.monotonic() - started_at) * 1000
        if played_ms >= total_ms:
            return
        try:
            await self._send_event({
                "type": "conversation.item.truncate",
                "item_id": item_id,
                "content_index": 0,
                "audio_end_ms": max(0, int(played_ms)),
            })
        except websockets.exceptions.ConnectionClosed:
            pass

    def _handle_response_created(self, response: Dict[str, Any]):
        response_id = response.get("id")
        metadata = response.get("metadata") or {}
        if response_id:
            self._response_metadata[response_id] = metadata
        untagged = self.match_untagged_responses and not metadata
        if self._summary_awaiting_response and (metadata.get("type") == "ending_analysis" or untagged):
            self._summary_awaiting_response = False
            self._summary_response_id = response_id
            self._summary_text = []
        elif self._farewell_awaiting_response and (metadata.get("type") == "final_farewell" or untagged):
            self._farewell_response_id = response_id
            self._farewell_awaiting_response = False

    async def _handle_response_done(self, response: Dict[str, Any]):
        response_id = response.get("id")
        metadata = self._response_metadata.pop(response_id, None) or response.get("metadata") or {}
        status = response.get("status")
        if self._audio_assembler:
            self._audio_assembler.flush()
        self._accumulate_usage(response.get("usage") or {})
        self.logger.debug(f"response.done id={response_id} status={status}")

        if metadata.get("type") == "ending_analysis" or (response_id and response_id == self._summary_response_id):
            text = self._extract_response_text(response) or "".join(self._summary_text)
            self._summary_response_id = None
            self._summary_text = []
            self._resolve_summary(text.strip() or None)
            return

        if (response_id and response_id == self._farewell_response_id) or metadata.get("type") == "final_farewell":
            self._spawn(self._finish_farewell(f"farewell {status or 'done'}"))
            return

        calls = self._collect_function_calls(response_id, response.get("output") or [])
        if calls:
            self._spawn(self._process_function_calls(calls))

    def _handle_error_event(self, event: Dict[str, Any]):
        error = event.get("error") or {}
        code = error.get("code") or event.get("code")
        message = error.get("message") or event.get("message")
        if code in _IGNORED_ERROR_CODES:
            self.logger.debug(f"{self.vendor_label} notice ({code}): {message}")
            return
        self.logger.warning(
            f"{self.vendor_label} error: type={error.get('type')} code={code} param={error.get('param')} message={message}"
        )

    def _log_mcp_event(self, event_type: str, event: Dict[str, Any]):
        item_id = event.get("item_id")
        if event_type.endswith("failed"):
            self.logger.warning(f"[MCP] {event_type} item={item_id}")
        elif event_type.endswith(("completed", "in_progress")):
            self.logger.info(f"[MCP] {event_type} item={item_id}")
        else:
            self.logger.debug(f"[MCP] {event_type} item={item_id}")

    def _accumulate_usage(self, usage: Dict[str, Any]):
        if not usage:
            return
        input_details = usage.get("input_token_details") or {}
        output_details = usage.get("output_token_details") or {}
        if not input_details and not output_details:
            self.usage.add(
                input_audio_tokens=usage.get("input_tokens"),
                output_audio_tokens=usage.get("output_tokens"),
            )
            return
        cached = input_details.get("cached_tokens_details") or {}
        self.usage.add(
            input_text_tokens=input_details.get("text_tokens"),
            input_audio_tokens=input_details.get("audio_tokens"),
            input_cached_text_tokens=cached.get("text_tokens"),
            input_cached_audio_tokens=cached.get("audio_tokens"),
            output_text_tokens=output_details.get("text_tokens"),
            output_audio_tokens=output_details.get("audio_tokens"),
        )

    @staticmethod
    def _extract_response_text(response: Dict[str, Any]) -> str:
        texts, transcripts = [], []
        for item in response.get("output") or []:
            if item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if part.get("type") in ("output_text", "text") and part.get("text"):
                    texts.append(part["text"])
                elif part.get("transcript"):
                    transcripts.append(part["transcript"])
        return "".join(texts) or "".join(transcripts)

    # ------------------------------------------------------------------ function calling

    def _record_function_call(self, response_id, call_id, name, arguments):
        if not call_id or not name or call_id in self._handled_call_ids:
            return
        bucket = self._pending_calls.setdefault(response_id or "", {})
        bucket.setdefault(call_id, PendingFunctionCall(call_id=call_id, name=name, arguments=arguments))

    def _collect_function_calls(self, response_id, output_items) -> List[PendingFunctionCall]:
        collected: Dict[str, PendingFunctionCall] = {}
        for key in {response_id or "", ""}:
            collected.update(self._pending_calls.pop(key, {}))
        for item in output_items:
            if item.get("type") == "function_call" and item.get("call_id") and item.get("name"):
                collected.setdefault(
                    item["call_id"],
                    PendingFunctionCall(call_id=item["call_id"], name=item["name"], arguments=item.get("arguments")),
                )
        calls = [call for call_id, call in collected.items() if call_id not in self._handled_call_ids]
        self._handled_call_ids.update(call.call_id for call in calls)
        return calls

    async def _process_function_calls(self, calls: List[PendingFunctionCall]):
        if self._disconnect_triggered or self._farewell is not None:
            return
        control_calls = [call for call in calls if is_call_control(call.name)]
        tool_calls = [call for call in calls if not is_call_control(call.name)]
        self.logger.info(f"Function calls received: {', '.join(call.name for call in calls)}")

        if tool_calls:
            outputs = await asyncio.gather(*(self._run_tool(call) for call in tool_calls))
            for call, output in zip(tool_calls, outputs):
                await self._send_function_output(call.call_id, output)

        if control_calls:
            first = control_calls[0]
            try:
                args = first.parsed_arguments()
            except ValueError:
                args = {}
            resolution = resolve_call_control(first.name, args, self.success_prompt, self.escalation_prompt)
            await self._send_function_output(first.call_id, resolution.output_payload)
            for extra in control_calls[1:]:
                await self._send_function_output(extra.call_id, {"result": "ignored", "reason": "call is already ending"})
            await self._begin_farewell(resolution)
            return

        await self._continue_after_tool_outputs()

    async def _continue_after_tool_outputs(self):
        await self._wait_for_playback()
        await self._send_event({"type": "response.create"})

    async def _send_function_output(self, call_id: str, payload: Dict[str, Any]):
        await self._send_event({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(payload, ensure_ascii=False, default=str),
            },
        })

    # ------------------------------------------------------------------ call ending

    async def _begin_farewell(self, resolution: CallControlResolution, wait_for_playback: bool = True):
        self._farewell_awaiting_response = True
        await super()._begin_farewell(resolution, wait_for_playback=wait_for_playback)

    # ------------------------------------------------------------------ summary

    async def generate_summary(self, prompt: str, timeout: float = 10.0) -> Optional[str]:
        if not self.running or not is_websocket_open(self.ws):
            return None
        self._summary_future = asyncio.get_running_loop().create_future()
        self._summary_awaiting_response = True
        try:
            await self._send_summary_request(prompt)
            return await asyncio.wait_for(self._summary_future, timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.warning("Timed out waiting for the end-of-call summary")
            return None
        except websockets.exceptions.ConnectionClosed:
            return None
        finally:
            self._summary_future = None
            self._summary_awaiting_response = False
