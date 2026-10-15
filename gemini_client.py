"""
Gemini Live API client (``gemini-3.8-live`` family, google-genai SDK) for the Genesys AudioHook bridge.

Caller audio is sent as 8 kHz PCM16 (``audio/pcm;rate=8000``; the Live API resamples input itself).
Model audio arrives as 24 kHz PCM16 and is converted to 8 kHz u-law with a stateful band-limited resampler.
Connections are reset by Google roughly every 10 minutes; the client resumes them transparently with
session resumption handles, and sliding-window context compression removes the 15-minute session limit.
"""

import asyncio
import audioop
import contextlib
import copy
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_VOICE,
    GEMINI_BASE_URL,
    GEMINI_API_VERSION,
    GEMINI_THINKING_LEVEL,
    GEMINI_VAD_SILENCE_DURATION_MS,
    GEMINI_VAD_PREFIX_PADDING_MS,
    GEMINI_VAD_START_SENSITIVITY,
    GEMINI_VAD_END_SENSITIVITY,
    GEMINI_CONTEXT_COMPRESSION,
    GEMINI_SESSION_RESUMPTION,
    GEMINI_GOOGLE_SEARCH,
    GEMINI_SUMMARY_MODEL,
)
from call_control import (
    CallControlResolution,
    call_control_function_tools,
    is_call_control,
    resolve_call_control,
)
from utils import Pcm16Resampler, PcmuFrameAssembler, create_final_system_prompt
from voice_client_base import BaseVoiceClient, PendingFunctionCall

GEMINI_OUTPUT_SAMPLE_RATE = 24000
GENESYS_SAMPLE_RATE = 8000
GEMINI_INPUT_MIME_TYPE = "audio/pcm;rate=8000"
CONNECT_TIMEOUT_SECONDS = 15.0
FAREWELL_QUIET_SECONDS = 1.5
_DEFAULT_GEMINI_HOST = "generativelanguage.googleapis.com"
_SEARCH_TOOL_TYPES = ("google_search", "web_search", "web_search_preview")

GEMINI_VOICES = (
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe",
    "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia",
    "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi", "Vindemiatrix",
    "Sadachbia", "Sadaltager", "Sulafat",
)


@dataclass(frozen=True)
class GeminiLiveProfile:
    """What a Live model accepts; sending unsupported settings fails the session setup."""
    thinking_levels: Tuple[str, ...] = ()
    non_blocking_tools: bool = False
    blocking_declarations: bool = False
    response_scheduling: bool = False
    # turn_complete does not mean idle while background reasoning runs; wait for interaction_status IDLE.
    idle_status_signal: bool = False


def gemini_live_profile(model: str) -> GeminiLiveProfile:
    name = (model or "").lower()
    if name.startswith("models/"):
        name = name[len("models/"):]
    if "extended-thinking" in name:
        return GeminiLiveProfile(thinking_levels=("low", "medium", "high"), non_blocking_tools=True,
                                 idle_status_signal=True)
    if name.startswith("gemini-3.1-flash-live"):
        return GeminiLiveProfile(thinking_levels=("minimal", "low", "medium", "high"))
    if name.startswith("gemini-2."):
        return GeminiLiveProfile()
    return GeminiLiveProfile(non_blocking_tools=True, blocking_declarations=True, response_scheduling=True)


def _clean_schema_for_gemini(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_clean_schema_for_gemini(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    cleaned = {}
    for key, value in schema.items():
        if key in ("strict", "additionalProperties", "$schema", "title"):
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {prop: _clean_schema_for_gemini(prop_schema) for prop, prop_schema in value.items()}
        else:
            cleaned[key] = _clean_schema_for_gemini(value)
    return cleaned


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


class GeminiRealtimeClient(BaseVoiceClient):
    vendor = "gemini"
    vendor_label = "Gemini"
    api_key_env_name = "GEMINI_API_KEY"
    default_model = GEMINI_MODEL
    default_voice = GEMINI_VOICE

    def __init__(self, session_id: str, api_key: Optional[str] = None, on_speech_started_callback=None):
        super().__init__(session_id, on_speech_started_callback=on_speech_started_callback, api_key=api_key)
        self.profile = gemini_live_profile(self.model)
        self.client: Optional[genai.Client] = None
        self.session = None
        self.receive_task: Optional[asyncio.Task] = None
        self._exit_stack: Optional[contextlib.AsyncExitStack] = None
        self._live_config: Optional[types.LiveConnectConfig] = None
        self._resumption_handle: Optional[str] = None
        self._reconnect_lock = asyncio.Lock()
        self._reconnecting = False
        self._go_away_handled = False
        self._send_lock = asyncio.Lock()
        self._function_count = 0

        self._on_audio_callback = None
        self._audio_assembler: Optional[PcmuFrameAssembler] = None
        self._output_resampler = Pcm16Resampler(GEMINI_OUTPUT_SAMPLE_RATE, GENESYS_SAMPLE_RATE)
        self._last_model_audio_at: Optional[float] = None

        self._transcript: List[Tuple[str, str]] = []
        self._cancelled_call_ids = set()
        self._farewell_sent_at: Optional[float] = None
        self._farewell_audio_seen = False

    def _default_api_key(self) -> Optional[str]:
        return GEMINI_API_KEY

    # ------------------------------------------------------------------ setup

    def _create_sdk_client(self) -> genai.Client:
        http_options: Dict[str, Any] = {"api_version": GEMINI_API_VERSION}
        host = (GEMINI_BASE_URL or "").strip().rstrip("/")
        if host and host != _DEFAULT_GEMINI_HOST and host != f"https://{_DEFAULT_GEMINI_HOST}":
            http_options["base_url"] = host if "://" in host else f"https://{host}"
        return genai.Client(api_key=self.api_key, http_options=types.HttpOptions(**http_options))

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
        self.profile = gemini_live_profile(self.model)
        self.voice = (voice or "").strip() or self.default_voice
        if self.voice not in GEMINI_VOICES:
            self.logger.warning(f"Voice '{self.voice}' is not a known Gemini Live voice; the session may reject it")
        if tool_choice not in (None, "auto"):
            self.logger.info("Gemini Live has no tool_choice setting; the model decides when to call functions")

        prompt = create_final_system_prompt(
            instructions,
            language=self.language,
            customer_data=self.customer_data,
            agent_name=agent_name,
            company_name=company_name,
        )
        if tool_instructions:
            prompt = f"{prompt}\n{tool_instructions}"

        self._live_config = self._build_live_config(prompt, tool_definitions, temperature, max_output_tokens)
        self.client = self._create_sdk_client()
        try:
            await self._open_session()
        except Exception as exc:
            await self.close()
            raise RuntimeError(f"Failed to connect to Gemini: {type(exc).__name__}: {exc}") from exc
        self.logger.info(
            f"Gemini session ready: model={self.model} voice={self.voice} "
            f"thinking={self._live_config.thinking_config is not None} tools={self._function_count}"
        )

    def _build_live_config(self, prompt, tool_definitions, temperature, max_output_tokens) -> types.LiveConnectConfig:
        vad_kwargs: Dict[str, Any] = {
            "silence_duration_ms": GEMINI_VAD_SILENCE_DURATION_MS,
            "prefix_padding_ms": GEMINI_VAD_PREFIX_PADDING_MS,
        }
        start = (GEMINI_VAD_START_SENSITIVITY or "").strip().upper()
        if start in ("HIGH", "LOW"):
            vad_kwargs["start_of_speech_sensitivity"] = types.StartSensitivity(f"START_SENSITIVITY_{start}")
        end = (GEMINI_VAD_END_SENSITIVITY or "").strip().upper()
        if end in ("HIGH", "LOW"):
            vad_kwargs["end_of_speech_sensitivity"] = types.EndSensitivity(f"END_SENSITIVITY_{end}")

        config_kwargs: Dict[str, Any] = {
            "response_modalities": [types.Modality.AUDIO],
            "system_instruction": prompt,
            "speech_config": types.SpeechConfig(
                voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice))
            ),
            "input_audio_transcription": types.AudioTranscriptionConfig(),
            "output_audio_transcription": types.AudioTranscriptionConfig(),
            "realtime_input_config": types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(**vad_kwargs)
            ),
            "tools": self._build_tools(tool_definitions),
        }
        if GEMINI_CONTEXT_COMPRESSION:
            config_kwargs["context_window_compression"] = types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow()
            )
        if GEMINI_SESSION_RESUMPTION:
            config_kwargs["session_resumption"] = types.SessionResumptionConfig(handle=None)

        level = (GEMINI_THINKING_LEVEL or "").strip().lower()
        if level and level in self.profile.thinking_levels:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=types.ThinkingLevel(level.upper()))
        elif level:
            self.logger.warning(f"GEMINI_THINKING_LEVEL={level} is not supported by {self.model}; omitting it")

        if temperature is not None:
            try:
                config_kwargs["temperature"] = float(temperature)
            except (TypeError, ValueError):
                pass
        if max_output_tokens not in (None, "", "inf"):
            try:
                tokens = int(max_output_tokens)
                if tokens > 0:
                    config_kwargs["max_output_tokens"] = tokens
            except (TypeError, ValueError):
                pass
        return types.LiveConnectConfig(**config_kwargs)

    def _build_tools(self, custom_tools: Optional[Iterable[Dict[str, Any]]]) -> List[types.Tool]:
        custom_tools = list(custom_tools or [])
        declarations = []
        for tool in call_control_function_tools() + [t for t in custom_tools if t.get("type") == "function"]:
            source = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            kwargs: Dict[str, Any] = {
                "name": source["name"],
                "description": source.get("description", ""),
                "parameters_json_schema": _clean_schema_for_gemini(
                    copy.deepcopy(source.get("parameters") or {"type": "object", "properties": {}})
                ),
            }
            if self.profile.non_blocking_tools and not self.profile.blocking_declarations:
                kwargs["behavior"] = types.Behavior.NON_BLOCKING
            elif self.profile.blocking_declarations and is_call_control(source["name"]):
                kwargs["behavior"] = types.Behavior.BLOCKING
            declarations.append(types.FunctionDeclaration(**kwargs))
        self._function_count = len(declarations)

        tools = [types.Tool(function_declarations=declarations)]
        if GEMINI_GOOGLE_SEARCH or any(t.get("type") in _SEARCH_TOOL_TYPES for t in custom_tools):
            tools.append(types.Tool(google_search=types.GoogleSearch()))
        skipped = sorted({t.get("type") for t in custom_tools if t.get("type") not in ("function", *_SEARCH_TOOL_TYPES)})
        if skipped:
            self.logger.warning(f"Skipping tool types not supported by Gemini Live: {', '.join(map(str, skipped))}")
        return tools

    async def _open_session(self):
        config = self._live_config
        if GEMINI_SESSION_RESUMPTION and self._resumption_handle:
            config = config.model_copy(
                update={"session_resumption": types.SessionResumptionConfig(handle=self._resumption_handle)}
            )
        stack = contextlib.AsyncExitStack()
        try:
            session = await asyncio.wait_for(
                stack.enter_async_context(self.client.aio.live.connect(model=self.model, config=config)),
                timeout=CONNECT_TIMEOUT_SECONDS,
            )
        except BaseException:
            await stack.aclose()
            raise
        self.session = session
        self._exit_stack = stack
        self._go_away_handled = False
        self.running = True

    async def _reconnect(self, reason: str) -> bool:
        if self._closing or not GEMINI_SESSION_RESUMPTION or not self._resumption_handle:
            return False
        async with self._reconnect_lock:
            if self._closing:
                return False
            old_stack = self._exit_stack
            self._reconnecting = True
            try:
                await self._open_session()
            except Exception as exc:
                self.logger.error(f"Gemini session resumption failed after {reason}: {type(exc).__name__}: {exc}")
                return False
            finally:
                self._reconnecting = False
            self.logger.info(f"Gemini session resumed after {reason}")
            if old_stack is not None and old_stack is not self._exit_stack:
                self._spawn(self._close_stack(old_stack))
            return True

    @staticmethod
    async def _close_stack(stack: contextlib.AsyncExitStack):
        try:
            await stack.aclose()
        except Exception:
            pass

    # ------------------------------------------------------------------ audio in / out

    async def send_audio(self, pcmu_8k: bytes):
        session = self.session
        if not self.running or self._reconnecting or session is None or not pcmu_8k:
            return
        pcm16 = audioop.ulaw2lin(pcmu_8k, 2)
        try:
            async with self._send_lock:
                await session.send_realtime_input(audio=types.Blob(data=pcm16, mime_type=GEMINI_INPUT_MIME_TYPE))
        except Exception as exc:
            if not self._reconnecting and session is self.session:
                self.logger.debug(f"Dropped caller audio: {type(exc).__name__}")

    async def start_receiving(self, on_audio_callback):
        self._on_audio_callback = on_audio_callback
        self._audio_assembler = PcmuFrameAssembler(self._emit_audio_frame)
        self.receive_task = asyncio.create_task(self._receive_loop())

    def _emit_audio_frame(self, frame: bytes):
        if self._on_audio_callback:
            self._on_audio_callback(frame)

    def _handle_model_audio(self, pcm16_24k: bytes):
        if not pcm16_24k or self._audio_assembler is None:
            return
        self._last_model_audio_at = time.monotonic()
        if self._farewell_sent_at is not None:
            self._farewell_audio_seen = True
        pcm16_8k = self._output_resampler.process(pcm16_24k)
        if pcm16_8k:
            self._audio_assembler.push(audioop.lin2ulaw(pcm16_8k, 2))

    def _flush_model_audio(self):
        if self._audio_assembler is None:
            return
        tail = self._output_resampler.process(b"", last=True)
        if tail:
            self._audio_assembler.push(audioop.lin2ulaw(tail, 2))
        self._output_resampler.reset()
        self._audio_assembler.flush()

    async def _handle_interrupted(self):
        if self._audio_assembler:
            self._audio_assembler.clear()
        self._output_resampler.reset()
        if self.on_speech_started_callback:
            try:
                await self.on_speech_started_callback()
            except Exception as exc:
                self.logger.error(f"Barge-in callback failed: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ receiving

    async def _receive_loop(self):
        try:
            while self.running and not self._closing:
                session = self.session
                if session is None:
                    break
                try:
                    # receive() stops after each completed model turn; keep listening for the next one.
                    async for message in session.receive():
                        try:
                            await self._handle_message(message)
                        except Exception as exc:
                            self.logger.error(f"Error handling Gemini message: {type(exc).__name__}: {exc}", exc_info=True)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if self._closing:
                        break
                    if self._reconnecting:
                        async with self._reconnect_lock:
                            pass
                    if session is not self.session:
                        continue
                    if await self._reconnect(f"connection closed ({type(exc).__name__})"):
                        continue
                    self.logger.warning(f"Gemini receive loop ended: {type(exc).__name__}: {exc}")
                    break
        finally:
            self.running = False
            self._resolve_summary(None)
            self._notify_connection_lost()

    async def _handle_message(self, message: types.LiveServerMessage):
        if message.session_resumption_update:
            update = message.session_resumption_update
            if update.resumable and update.new_handle:
                self._resumption_handle = update.new_handle

        if message.go_away and not self._go_away_handled:
            self._go_away_handled = True
            self.logger.info(f"Gemini GoAway received (time_left={message.go_away.time_left})")
            self._spawn(self._reconnect("GoAway"))

        if message.usage_metadata:
            self._accumulate_usage(message.usage_metadata)

        if message.tool_call and message.tool_call.function_calls:
            self._spawn(self._handle_function_calls(list(message.tool_call.function_calls)))

        if message.tool_call_cancellation and message.tool_call_cancellation.ids:
            self._cancelled_call_ids.update(message.tool_call_cancellation.ids)
            self.logger.info(f"Gemini cancelled {len(message.tool_call_cancellation.ids)} pending function call(s)")

        content = message.server_content
        if content is None:
            return
        if content.interrupted:
            await self._handle_interrupted()
        if content.model_turn and content.model_turn.parts:
            for part in content.model_turn.parts:
                if part.inline_data and part.inline_data.data:
                    self._handle_model_audio(part.inline_data.data)
        if content.input_transcription and content.input_transcription.text:
            self._append_transcript("Caller", content.input_transcription.text)
        if content.output_transcription and content.output_transcription.text:
            self._append_transcript("Agent", content.output_transcription.text)
        if content.generation_complete or content.turn_complete:
            self._flush_model_audio()
            if self._farewell_audio_seen and not self.profile.idle_status_signal:
                self._spawn(self._finish_farewell("farewell spoken"))
        if (self.profile.idle_status_signal and self._farewell_audio_seen
                and _enum_value(content.interaction_status) == "IDLE"):
            self._spawn(self._finish_farewell("farewell spoken"))
        if content.turn_complete_reason and _enum_value(content.turn_complete_reason) not in (
            "TURN_COMPLETE_REASON_UNSPECIFIED", "NEED_MORE_INPUT"
        ):
            self.logger.warning(f"Gemini turn ended early: {_enum_value(content.turn_complete_reason)}")

    def _append_transcript(self, speaker: str, text: str):
        if self._transcript and self._transcript[-1][0] == speaker:
            self._transcript[-1] = (speaker, self._transcript[-1][1] + text)
        else:
            self._transcript.append((speaker, text))

    def _accumulate_usage(self, usage: types.UsageMetadata):
        def by_modality(details) -> Dict[str, int]:
            totals: Dict[str, int] = {}
            for detail in details or []:
                modality = str(_enum_value(detail.modality))
                totals[modality] = totals.get(modality, 0) + (detail.token_count or 0)
            return totals

        prompt = by_modality(usage.prompt_tokens_details)
        cached = by_modality(usage.cache_tokens_details)
        response = by_modality(usage.response_tokens_details)
        self.usage.add(
            input_text_tokens=prompt.get("TEXT"),
            input_audio_tokens=prompt.get("AUDIO") if prompt else usage.prompt_token_count,
            input_cached_text_tokens=cached.get("TEXT"),
            input_cached_audio_tokens=cached.get("AUDIO"),
            output_text_tokens=(response.get("TEXT") or 0) + (usage.thoughts_token_count or 0),
            output_audio_tokens=response.get("AUDIO") if response else usage.response_token_count,
        )

    # ------------------------------------------------------------------ function calling

    async def _handle_function_calls(self, function_calls: List[types.FunctionCall]):
        if self._disconnect_triggered:
            return
        calls = [PendingFunctionCall(call_id=fc.id, name=fc.name, arguments=fc.args or {}) for fc in function_calls]
        self.logger.info(f"Function calls received: {', '.join(call.name for call in calls)}")
        control_calls = [call for call in calls if is_call_control(call.name)]
        tool_calls = [call for call in calls if not is_call_control(call.name)]

        responses: List[types.FunctionResponse] = []
        if tool_calls:
            outputs = await asyncio.gather(*(self._run_tool(call) for call in tool_calls))
            for call, output in zip(tool_calls, outputs):
                if call.call_id in self._cancelled_call_ids:
                    continue
                responses.append(self._function_response(call, output, scheduling="WHEN_IDLE"))

        resolution: Optional[CallControlResolution] = None
        if control_calls and self._farewell is None:
            first = control_calls[0]
            try:
                args = first.parsed_arguments()
            except ValueError:
                args = {}
            resolution = resolve_call_control(first.name, args, self.success_prompt, self.escalation_prompt)
            payload = {**resolution.output_payload, "instruction": resolution.farewell_instruction}
            responses.append(self._function_response(first, payload, scheduling="INTERRUPT", blocking=True))
            for extra in control_calls[1:]:
                responses.append(self._function_response(extra, {"result": "ignored", "reason": "call is already ending"}))

        if resolution is not None:
            self._farewell_sent_at = time.monotonic()
            self._farewell_audio_seen = False
        if responses and self.session is not None:
            async with self._send_lock:
                await self.session.send_tool_response(function_responses=responses)
        if resolution is not None:
            await self._begin_farewell(resolution, wait_for_playback=False)

    def _function_response(self, call: PendingFunctionCall, payload: Dict[str, Any],
                           scheduling: Optional[str] = None, blocking: bool = False) -> types.FunctionResponse:
        response = dict(payload)
        if scheduling and self.profile.response_scheduling and not (blocking and self.profile.blocking_declarations):
            response["scheduling"] = scheduling
        return types.FunctionResponse(id=call.call_id, name=call.name, response=response)

    async def _send_farewell(self, resolution: CallControlResolution):
        if not self.profile.idle_status_signal:
            self._spawn(self._await_farewell_spoken())

    async def _await_farewell_spoken(self):
        while not self._disconnect_triggered and self.running:
            await asyncio.sleep(0.1)
            last_audio = self._last_model_audio_at
            if (self._farewell_audio_seen and last_audio is not None
                    and time.monotonic() - last_audio >= FAREWELL_QUIET_SECONDS):
                await self._finish_farewell("farewell audio finished")
                return

    # ------------------------------------------------------------------ summary & close

    async def generate_summary(self, prompt: str, timeout: float = 10.0) -> Optional[str]:
        transcript = "\n".join(f"{speaker}: {text.strip()}" for speaker, text in self._transcript if text.strip())
        if not transcript or self.client is None:
            return None
        try:
            response = await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model=GEMINI_SUMMARY_MODEL,
                    contents=f"Conversation transcript:\n{transcript}",
                    config=types.GenerateContentConfig(system_instruction=prompt, max_output_tokens=600),
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            self.logger.warning("Timed out waiting for the end-of-call summary")
            return None
        except Exception as exc:
            self.logger.warning(f"Summary request failed: {type(exc).__name__}")
            return None
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            self.usage.add(input_text_tokens=usage.prompt_token_count, output_text_tokens=usage.candidates_token_count)
        text = getattr(response, "text", None)
        return text.strip() if text else None

    async def close(self):
        self._closing = True
        self.running = False
        self._cancel_farewell_timeout()
        self._cancel_background_tasks()
        stack, self._exit_stack = self._exit_stack, None
        self.session = None
        if stack is not None:
            await self._close_stack(stack)
        if self.receive_task and self.receive_task is not asyncio.current_task() and not self.receive_task.done():
            self.receive_task.cancel()
            try:
                await self.receive_task
            except (asyncio.CancelledError, Exception):
                pass
        self._resolve_summary(None)
