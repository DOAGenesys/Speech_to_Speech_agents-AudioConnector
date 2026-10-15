import asyncio
import uuid
import json
import time
import websockets

from config import (
    logger,
    RATE_LIMIT_MAX_RETRIES,
    RATE_LIMIT_PHASES,
    GENESYS_MSG_RATE_LIMIT,
    GENESYS_BINARY_RATE_LIMIT,
    GENESYS_MSG_BURST_LIMIT,
    GENESYS_BINARY_BURST_LIMIT,
    DEFAULT_AGENT_NAME,
    DEFAULT_COMPANY_NAME,
    DEFAULT_MAX_OUTPUT_TOKENS,
    MAX_AUDIO_BUFFER_SIZE,
    AUDIO_BUFFER_WARNING_THRESHOLD_HIGH,
    AUDIO_BUFFER_WARNING_THRESHOLD_MEDIUM,
    AI_VENDOR,
    VENDOR_API_KEY_ENV_NAMES,
    VENDOR_DEFAULT_MODELS,
    VENDOR_DEFAULT_VOICES,
    get_vendor_api_key,
    model_matches_vendor,
    normalize_ai_vendor,
)

from rate_limiter import RateLimiter
from openai_client import OpenAIRealtimeClient
from gpt_live_client import GptLiveClient, is_gpt_live_model
from grok_client import GrokRealtimeClient
from gemini_client import GEMINI_VOICES, GeminiRealtimeClient
from utils import format_json, parse_iso8601_duration
from genesys_actions import build_genesys_tool_context
from mcp_tools import load_mcp_tool_context

from collections import deque

PCMU_BYTES_PER_SECOND = 8000
DISCONNECT_PLAYBACK_WAIT_SECONDS = 12.0
SUMMARY_TIMEOUT_SECONDS = 10.0
CONVERSATION_SUMMARY_PROMPT = (
    "Please provide a brief, compact summary of this conversation in plain text (2-3 sentences). "
    "Do not use JSON format, just plain text."
)
VENDOR_LABELS = {"openai": "OpenAI", "gemini": "Gemini", "grok": "xAI Grok"}
VENDOR_MODEL_INPUT_VARS = {"openai": "OPENAI_MODEL", "gemini": "GEMINI_MODEL", "grok": "GROK_MODEL"}
VENDOR_VOICE_INPUT_VARS = {"openai": "OPENAI_VOICE", "gemini": "GEMINI_VOICE", "grok": "GROK_VOICE"}


class AudioHookServer:
    def __init__(self, websocket):
        """
        Manages one Genesys AudioHook websocket: protocol state, rate limits, the outbound audio buffer
        and the speech-to-speech vendor client for the call.
        """
        self.session_id = str(uuid.uuid4())
        self.ws = websocket
        self.client_seq = 0
        self.server_seq = 0
        self.ai_client = None
        self.ai_vendor = AI_VENDOR
        self.running = True
        self.negotiated_media = None
        self.start_time = time.time()
        self.logger = logger.getChild(f"AudioHookServer_{self.session_id}")
        self.audio_frames_sent = 0
        self.audio_frames_received = 0
        self.rate_limit_state = {
            "retry_count": 0,
            "last_retry_time": 0,
            "in_backoff": False
        }

        self.message_limiter = RateLimiter(GENESYS_MSG_RATE_LIMIT, GENESYS_MSG_BURST_LIMIT)
        self.binary_limiter = RateLimiter(GENESYS_BINARY_RATE_LIMIT, GENESYS_BINARY_BURST_LIMIT)

        self.audio_buffer = deque(maxlen=MAX_AUDIO_BUFFER_SIZE)
        self.audio_process_task = None
        self._playback_end_monotonic = 0.0
        self._disconnecting = False
        self.genesys_tool_context = None
        self.session_outcome = {
            "escalation_required": False,
            "escalation_reason": "",
            "completion_summary": "",
        }

        self.logger.info(f"New session started: {self.session_id}")

    # ------------------------------------------------------------------ outbound audio

    async def start_audio_processing(self):
        self.audio_process_task = asyncio.create_task(self._process_audio_buffer())

    async def stop_audio_processing(self):
        if self.audio_process_task:
            self.audio_process_task.cancel()
            try:
                await self.audio_process_task
            except asyncio.CancelledError:
                pass
            self.audio_process_task = None

    async def _process_audio_buffer(self):
        """
        Sends buffered model audio to Genesys as fast as the binary rate limiter allows.
        Genesys paces playback itself, so frames are not delayed artificially.
        """
        try:
            while self.running:
                if self.rate_limit_state["in_backoff"] or not self.audio_buffer:
                    await asyncio.sleep(0.01)
                    continue
                if not await self.binary_limiter.acquire():
                    await asyncio.sleep(0.01)
                    continue
                frame_bytes = self.audio_buffer.popleft()
                try:
                    await self.ws.send(frame_bytes)
                except websockets.ConnectionClosed:
                    self.logger.warning("Genesys WebSocket closed while sending audio frame.")
                    self.running = False
                    break
                self.audio_frames_sent += 1
                now = time.monotonic()
                self._playback_end_monotonic = max(now, self._playback_end_monotonic) + len(frame_bytes) / PCMU_BYTES_PER_SECOND
        except asyncio.CancelledError:
            self.logger.info("Audio processing task cancelled")
        except Exception as e:
            self.logger.error(f"Error in audio processing task: {e}", exc_info=True)

    def get_playback_remaining_seconds(self) -> float:
        """Seconds of model audio the caller has not heard yet (buffered here plus already sent to Genesys)."""
        buffered = sum(len(frame) for frame in self.audio_buffer) / PCMU_BYTES_PER_SECOND
        in_flight = max(0.0, self._playback_end_monotonic - time.monotonic())
        return buffered + in_flight

    def enqueue_ai_audio(self, pcmu_8k: bytes):
        if not self.running:
            return
        if len(self.audio_buffer) >= MAX_AUDIO_BUFFER_SIZE:
            self.logger.error(
                f"Audio buffer FULL ({len(self.audio_buffer)} frames), dropping frame - response may be truncated! "
                f"Consider increasing MAX_AUDIO_BUFFER_SIZE for longer responses."
            )
            return
        self.audio_buffer.append(pcmu_8k)
        buffer_usage = len(self.audio_buffer) / MAX_AUDIO_BUFFER_SIZE
        if buffer_usage >= AUDIO_BUFFER_WARNING_THRESHOLD_HIGH:
            self.logger.warning(
                f"Audio buffer usage HIGH: {len(self.audio_buffer)}/{MAX_AUDIO_BUFFER_SIZE} frames "
                f"({buffer_usage*100:.1f}%) - Long response in progress"
            )
        elif buffer_usage >= AUDIO_BUFFER_WARNING_THRESHOLD_MEDIUM:
            self.logger.info(
                f"Audio buffer usage elevated: {len(self.audio_buffer)}/{MAX_AUDIO_BUFFER_SIZE} frames "
                f"({buffer_usage*100:.1f}%)"
            )

    # ------------------------------------------------------------------ Genesys protocol

    async def handle_error(self, msg: dict):
        error_code = msg["parameters"].get("code")
        error_params = msg["parameters"]

        if error_code == 429:
            retry_after = None

            if "retryAfter" in error_params:
                retry_after_duration = error_params["retryAfter"]
                try:
                    retry_after = parse_iso8601_duration(retry_after_duration)
                    self.logger.info(
                        f"[Rate Limit] Using Genesys-provided retryAfter duration: {retry_after}s "
                        f"(parsed from {retry_after_duration})"
                    )
                except ValueError as e:
                    self.logger.warning(
                        f"[Rate Limit] Failed to parse Genesys retryAfter format: {retry_after_duration}. "
                        f"Error: {str(e)}"
                    )

            if retry_after is None and hasattr(self.ws, 'response_headers'):
                http_retry_after = (
                    self.ws.response_headers.get('Retry-After') or
                    self.ws.response_headers.get('retry-after')
                )
                if http_retry_after:
                    try:
                        retry_after = float(http_retry_after)
                        self.logger.info(
                            f"[Rate Limit] Using HTTP header Retry-After duration: {retry_after}s"
                        )
                    except ValueError:
                        try:
                            retry_after = parse_iso8601_duration(http_retry_after)
                            self.logger.info(
                                f"[Rate Limit] Using HTTP header Retry-After duration: {retry_after}s "
                                f"(parsed from ISO8601)"
                            )
                        except ValueError:
                            self.logger.warning(
                                f"[Rate Limit] Failed to parse HTTP Retry-After format: {http_retry_after}"
                            )

            self.logger.warning(
                f"[Rate Limit] Received 429 error. "
                f"Session: {self.session_id}, "
                f"Current duration: {time.time() - self.start_time:.2f}s, "
                f"Retry count: {self.rate_limit_state['retry_count']}, "
                f"RetryAfter: {retry_after}s"
            )

            self.rate_limit_state["in_backoff"] = True
            self.rate_limit_state["retry_count"] += 1

            if self.rate_limit_state["retry_count"] > RATE_LIMIT_MAX_RETRIES:
                self.logger.error(
                    f"[Rate Limit] Max retries ({RATE_LIMIT_MAX_RETRIES}) exceeded. "
                    f"Session: {self.session_id}, "
                    f"Total retries: {self.rate_limit_state['retry_count']}, "
                    f"Duration: {time.time() - self.start_time:.2f}s"
                )
                self.rate_limit_state["in_backoff"] = False
                await self.disconnect_session(reason="error", info="Rate limit max retries exceeded")
                return False

            session_duration = time.time() - self.start_time

            if retry_after is not None:
                used_delay = retry_after
                self.logger.info(
                    f"[Rate Limit] Using provided retry delay: {used_delay}s. "
                    f"Session: {self.session_id}"
                )
            else:
                delay = None
                for phase in RATE_LIMIT_PHASES:
                    if session_duration <= phase["window"]:
                        delay = phase["delay"]
                        break
                if delay is None:
                    delay = RATE_LIMIT_PHASES[-1]["delay"]
                used_delay = delay
                self.logger.info(
                    f"[Rate Limit] Using default exponential backoff delay: {used_delay}s. "
                    f"Session: {self.session_id}"
                )

            self.logger.warning(
                f"[Rate Limit] Rate limited, attempt {self.rate_limit_state['retry_count']}/{RATE_LIMIT_MAX_RETRIES}. "
                f"Backing off for {used_delay}s. "
                f"Session: {self.session_id}, "
                f"Duration: {session_duration:.2f}s"
            )

            await asyncio.sleep(used_delay)

            self.rate_limit_state["in_backoff"] = False
            self.logger.info(
                f"[Rate Limit] Backoff complete, resuming operations. "
                f"Session: {self.session_id}"
            )

            return True
        return False

    async def handle_message(self, msg: dict):
        msg_type = msg.get("type")
        seq = msg.get("seq", 0)
        self.client_seq = seq

        if self.rate_limit_state.get("in_backoff") and msg_type != "error":
            self.logger.debug(f"Skipping message type {msg_type} during rate limit backoff")
            return

        if msg_type == "error":
            handled = await self.handle_error(msg)
            if handled:
                return

        if msg_type == "open":
            await self.handle_open(msg)
        elif msg_type == "ping":
            await self.handle_ping(msg)
        elif msg_type == "close":
            await self.handle_close(msg)
        elif msg_type in ["update", "resume", "pause"]:
            self.logger.debug(f"Ignoring message of type {msg_type}")
        else:
            self.logger.debug(f"Ignoring unknown message type: {msg_type}")

    def _select_vendor(self, input_vars: dict):
        """Return (vendor, error). AI_VENDOR in the call's input variables overrides the deployment default."""
        requested = input_vars.get("AI_VENDOR")
        if not requested or not str(requested).strip():
            return AI_VENDOR, None
        vendor = normalize_ai_vendor(requested)
        if vendor is None:
            self.logger.warning(f"Unsupported AI_VENDOR input variable '{requested}'; using default vendor {AI_VENDOR}")
            return AI_VENDOR, None
        if not get_vendor_api_key(vendor):
            return None, f"AI_VENDOR={vendor} requested but {VENDOR_API_KEY_ENV_NAMES[vendor]} is not configured on the server"
        return vendor, None

    def _select_model(self, vendor: str, input_vars: dict) -> str:
        candidates = [
            (VENDOR_MODEL_INPUT_VARS[vendor], input_vars.get(VENDOR_MODEL_INPUT_VARS[vendor])),
            ("AI_MODEL", input_vars.get("AI_MODEL")),
        ]
        if vendor != "openai":
            # OPENAI_MODEL was the legacy name for AI_MODEL; only honour it when it names a model of this vendor.
            candidates.append(("OPENAI_MODEL", input_vars.get("OPENAI_MODEL")))
        for name, value in candidates:
            if not value or not str(value).strip():
                continue
            value = str(value).strip()
            if model_matches_vendor(vendor, value):
                return value
            log = self.logger.debug if name == "OPENAI_MODEL" else self.logger.warning
            log(f"Ignoring {name}='{value}': it is not a {VENDOR_LABELS[vendor]} model")
        return VENDOR_DEFAULT_MODELS[vendor]

    def _select_voice(self, vendor: str, input_vars: dict) -> str:
        voice = None
        for name in (VENDOR_VOICE_INPUT_VARS[vendor], "AI_VOICE"):
            value = input_vars.get(name)
            if value and str(value).strip():
                voice = str(value).strip()
                break
        voice = voice or VENDOR_DEFAULT_VOICES[vendor]
        if vendor == "gemini":
            canonical = {name.lower(): name for name in GEMINI_VOICES}.get(voice.lower())
            if canonical is None:
                self.logger.warning(f"'{voice}' is not a Gemini Live voice; using {VENDOR_DEFAULT_VOICES['gemini']}")
                return VENDOR_DEFAULT_VOICES["gemini"]
            return canonical
        return voice

    def _create_ai_client(self, vendor: str, model: str):
        if vendor == "gemini":
            return GeminiRealtimeClient(
                self.session_id,
                api_key=get_vendor_api_key("gemini"),
                on_speech_started_callback=self.handle_speech_started,
            )
        if vendor == "grok":
            return GrokRealtimeClient(
                self.session_id,
                on_speech_started_callback=self.handle_speech_started,
                api_key=get_vendor_api_key("grok"),
            )
        if is_gpt_live_model(model):
            return GptLiveClient(
                self.session_id,
                on_speech_started_callback=self.handle_speech_started,
                api_key=get_vendor_api_key("openai"),
            )
        return OpenAIRealtimeClient(
            self.session_id,
            on_speech_started_callback=self.handle_speech_started,
            api_key=get_vendor_api_key("openai"),
        )

    async def _send_disconnect_without_session(self, info: str):
        self.logger.error(info)
        resp = {
            "version": "2",
            "type": "disconnect",
            "seq": self.server_seq + 1,
            "clientseq": self.client_seq,
            "id": self.session_id,
            "parameters": {"reason": "error", "info": info}
        }
        self.server_seq += 1
        await self._send_json(resp)
        self.running = False

    async def handle_open(self, msg: dict):
        self.session_id = msg["id"]

        is_probe = (
            msg["parameters"].get("conversationId") == "00000000-0000-0000-0000-000000000000" and
            msg["parameters"].get("participant", {}).get("id") == "00000000-0000-0000-0000-000000000000"
        )

        if is_probe:
            self.logger.info("Detected probe connection")
            opened_msg = {
                "version": "2",
                "type": "opened",
                "seq": self.server_seq + 1,
                "clientseq": self.client_seq,
                "id": self.session_id,
                "parameters": {
                    "startPaused": False,
                    "media": []
                }
            }
            self.server_seq += 1
            await self._send_json(opened_msg)
            return

        offered_media = msg["parameters"].get("media", [])
        chosen = None
        for m in offered_media:
            if (m.get("format") == "PCMU" and m.get("rate") == 8000):
                chosen = m
                break

        if not chosen:
            await self._send_disconnect_without_session("No supported format found")
            return

        input_vars = msg["parameters"].get("inputVariables", {}) or {}
        vendor, vendor_error = self._select_vendor(input_vars)
        if vendor_error:
            await self._send_disconnect_without_session(vendor_error)
            return
        self.ai_vendor = vendor

        opened_msg = {
            "version": "2",
            "type": "opened",
            "seq": self.server_seq + 1,
            "clientseq": self.client_seq,
            "id": self.session_id,
            "parameters": {
                "startPaused": False,
                "media": [chosen]
            }
        }
        self.server_seq += 1
        await self._send_json(opened_msg)
        self.logger.info(f"Session opened. Negotiated media format: {chosen}")

        model = self._select_model(vendor, input_vars)
        voice = self._select_voice(vendor, input_vars)

        instructions = input_vars.get("AI_SYSTEM_PROMPT")
        if not instructions:
            # Fallback to legacy OPENAI_SYSTEM_PROMPT for backward compatibility
            instructions = input_vars.get("OPENAI_SYSTEM_PROMPT", "You are a helpful assistant.")

        temperature = input_vars.get("AI_TEMPERATURE")
        if not temperature:
            # Fallback to legacy OPENAI_TEMPERATURE for backward compatibility
            temperature = input_vars.get("OPENAI_TEMPERATURE")

        max_output_tokens = input_vars.get("AI_MAX_OUTPUT_TOKENS")
        if not max_output_tokens:
            # Fallback to legacy OPENAI_MAX_OUTPUT_TOKENS for backward compatibility
            max_output_tokens = input_vars.get("OPENAI_MAX_OUTPUT_TOKENS")

        language = input_vars.get("LANGUAGE")
        customer_data = input_vars.get("CUSTOMER_DATA")
        agent_name = input_vars.get("AGENT_NAME", DEFAULT_AGENT_NAME)
        company_name = next((value for key, value in input_vars.items()
                            if key.strip() == "COMPANY_NAME"), DEFAULT_COMPANY_NAME)
        escalation_prompt = input_vars.get("ESCALATION_PROMPT")
        success_prompt = input_vars.get("SUCCESS_PROMPT")

        try:
            self.genesys_tool_context = await build_genesys_tool_context(self.logger, input_vars)
        except Exception as exc:
            self.logger.error(f"[GenesysTools] Failed to prepare data action tools: {exc}")
            self.genesys_tool_context = None
        tool_definitions = []
        tool_instruction_blocks = []
        if self.genesys_tool_context:
            tool_definitions.extend(self.genesys_tool_context.tools)
            if self.genesys_tool_context.instructions:
                tool_instruction_blocks.append(self.genesys_tool_context.instructions)
            self.logger.info(f"[FunctionCall] Enabled {len(self.genesys_tool_context.tools)} Genesys data action tools for this session")

        mcp_raw_json = input_vars.get("MCP_TOOLS_JSON")
        mcp_context = load_mcp_tool_context(mcp_raw_json, self.logger)
        if mcp_context:
            tool_definitions.extend(mcp_context.tools)
            if mcp_context.instructions:
                tool_instruction_blocks.append(mcp_context.instructions)
            self.logger.info(f"[MCP] Enabled {len(mcp_context.tools)} MCP/built-in tool definitions for this session")

        tool_definitions_payload = tool_definitions if tool_definitions else None
        tool_instructions = "\n\n".join(block for block in tool_instruction_blocks if block) if tool_instruction_blocks else None

        self.logger.info(f"[AI Vendor] {VENDOR_LABELS[vendor]} model={model} voice={voice}")
        self.logger.debug(f"Using instructions: {instructions}")
        if temperature:
            self.logger.debug(f"Using temperature: {temperature}")
        if max_output_tokens:
            if str(max_output_tokens).lower() == 'inf':
                max_output_tokens = "inf"
            else:
                try:
                    tokens = int(max_output_tokens)
                    if 1 <= tokens <= 4096:
                        max_output_tokens = tokens
                    else:
                        self.logger.warning(f"max_output_tokens {tokens} out of range [1, 4096]. Using default: {DEFAULT_MAX_OUTPUT_TOKENS}")
                        max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS
                except (TypeError, ValueError):
                    self.logger.warning(f"Invalid max_output_tokens value: {max_output_tokens}. Using default: {DEFAULT_MAX_OUTPUT_TOKENS}")
                    max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS
        else:
            max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS
        if language:
            self.logger.info(f"Enforcing language: {language}")
        if customer_data:
            self.logger.debug("Customer data provided for personalization")
        if escalation_prompt:
            self.logger.info("Using custom ESCALATION_PROMPT")
        if success_prompt:
            self.logger.info("Using custom SUCCESS_PROMPT")

        try:
            client = self._create_ai_client(vendor, model)
            self.ai_client = client
            client.language = language
            client.customer_data = customer_data
            client.escalation_prompt = escalation_prompt
            client.success_prompt = success_prompt
            client.on_end_call_request = self._on_end_call_request
            client.on_handoff_request = self._on_handoff_request
            client.on_connection_lost = self._on_ai_connection_lost
            client.playback_remaining_fn = self.get_playback_remaining_seconds
            if isinstance(client, GrokRealtimeClient):
                client.set_keyterms(input_vars.get("GROK_KEYTERMS"))
                if input_vars.get("GROK_REASONING_EFFORT"):
                    client.reasoning_effort = input_vars.get("GROK_REASONING_EFFORT")
            if self.genesys_tool_context:
                client.register_genesys_tool_handlers(self.genesys_tool_context.handlers)

            await client.connect(
                instructions=instructions,
                voice=voice,
                temperature=temperature,
                model=model,
                max_output_tokens=max_output_tokens,
                agent_name=agent_name,
                company_name=company_name,
                tool_definitions=tool_definitions_payload,
                tool_instructions=tool_instructions
            )
        except Exception as e:
            self.logger.error(f"AI client connection failed: {e}")
            await self.disconnect_session(reason="error", info=str(e))
            return

        await self.start_audio_processing()
        await self.ai_client.start_receiving(self.enqueue_ai_audio)

    async def _on_end_call_request(self, reason: str, info: str):
        self.logger.info(f"[FunctionCall] Model ended the call. reason={reason}")
        self.session_outcome.update({
            "escalation_required": False,
            "escalation_reason": "",
            "completion_summary": info or "",
        })
        await self.disconnect_session(reason=reason or "completed", info=info or "")

    async def _on_handoff_request(self, reason: str, info: str):
        self.logger.info(f"[FunctionCall] Model requested escalation to a human agent. reason={reason}")
        escalation_reason = info or reason or "Customer requested escalation to agent"
        self.session_outcome.update({
            "escalation_required": True,
            "escalation_reason": escalation_reason
        })
        await self.disconnect_session(reason="completed", info=escalation_reason)

    async def _on_ai_connection_lost(self):
        if self._disconnecting or not self.running:
            return
        label = VENDOR_LABELS.get(self.ai_vendor, self.ai_vendor)
        self.logger.error(f"{label} connection lost mid-call; disconnecting the session")
        await self.disconnect_session(reason="error", info=f"{label} connection lost")

    async def handle_speech_started(self):
        self.audio_buffer.clear()
        self._playback_end_monotonic = time.monotonic()
        event_msg = {
            "version": "2",
            "type": "event",
            "seq": self.server_seq + 1,
            "clientseq": self.client_seq,
            "id": self.session_id,
            "parameters": {
                "entities": [
                    {
                        "type": "barge_in",
                        "data": {}
                    }
                ]
            }
        }
        self.server_seq += 1
        await self._send_json(event_msg)

    async def handle_ping(self, msg: dict):
        pong_msg = {
            "version": "2",
            "type": "pong",
            "seq": self.server_seq + 1,
            "clientseq": self.client_seq,
            "id": self.session_id,
            "parameters": {}
        }
        self.server_seq += 1
        try:
            await asyncio.wait_for(self._send_json(pong_msg), timeout=1.0)
        except asyncio.TimeoutError:
            self.logger.error("Failed to send pong response within timeout")

    async def generate_session_summary(self):
        if not self.ai_client:
            return None
        try:
            summary = await self.ai_client.generate_summary(CONVERSATION_SUMMARY_PROMPT, timeout=SUMMARY_TIMEOUT_SECONDS)
        except Exception as e:
            self.logger.error(f"Error generating session summary: {type(e).__name__}: {e}")
            return None
        return summary.strip() if summary else None

    async def handle_close(self, msg: dict):
        self.logger.info(f"Received 'close' from Genesys. Reason: {msg['parameters'].get('reason')}")
        self._disconnecting = True
        summary = await self.generate_session_summary()
        if summary:
            self.logger.info("Session summary generated")

        closed_msg = {
            "version": "2",
            "type": "closed",
            "seq": self.server_seq + 1,
            "clientseq": self.client_seq,
            "id": self.session_id,
            "parameters": {
                "summary": summary
            }
        }
        self.server_seq += 1
        try:
            await asyncio.wait_for(self._send_json(closed_msg), timeout=4.0)
        except asyncio.TimeoutError:
            self.logger.error("Failed to send closed response within timeout")

        duration = time.time() - self.start_time
        self.logger.info(
            f"Session stats - Duration: {duration:.2f}s, "
            f"Frames sent: {self.audio_frames_sent}, "
            f"Frames received: {self.audio_frames_received}"
        )

        await self.stop_audio_processing()
        if self.ai_client:
            await self.ai_client.close()
        self.running = False

    async def _wait_for_playback_to_finish(self, max_wait: float):
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline and self.running:
            if not self.audio_buffer and self.get_playback_remaining_seconds() <= 0.05:
                return True
            await asyncio.sleep(0.1)
        return not self.audio_buffer

    async def disconnect_session(self, reason="completed", info=""):
        if self._disconnecting:
            return
        self._disconnecting = True
        try:
            if not self.session_id:
                return

            self.logger.info(f"[FunctionCall] Initiating server-side disconnect. reason={reason}")

            # The summary runs while the farewell finishes playing so the caller does not wait for both.
            summary_task = asyncio.create_task(self.generate_session_summary()) if self.ai_client else None

            wait_start = time.time()
            drained = await self._wait_for_playback_to_finish(DISCONNECT_PLAYBACK_WAIT_SECONDS)
            if drained:
                self.logger.info(f"[FunctionCall] Farewell playback finished after {time.time() - wait_start:.2f}s")
            else:
                self.logger.warning(
                    f"[FunctionCall] Audio buffer still has {len(self.audio_buffer)} frames after "
                    f"{DISCONNECT_PLAYBACK_WAIT_SECONDS}s wait - proceeding with disconnect"
                )

            self.running = False
            await self.stop_audio_processing()

            summary_data = await summary_task if summary_task else None

            token_metrics = {}
            if self.ai_client:
                await self.ai_client.close()
                token_metrics = self.ai_client.get_token_metrics()
                self.logger.info(f"[FunctionCall] Token usage ({VENDOR_LABELS.get(self.ai_vendor)}): {token_metrics}")

            outcome = self.session_outcome or {}
            output_vars = {
                "CONVERSATION_SUMMARY": summary_data if summary_data else "",
                "CONVERSATION_DURATION": str(time.time() - self.start_time),
                "ESCALATION_REQUIRED": "true" if outcome.get("escalation_required") else "false",
                "ESCALATION_REASON": outcome.get("escalation_reason", ""),
                "COMPLETION_SUMMARY": outcome.get("completion_summary", ""),
                "TOTAL_INPUT_TEXT_TOKENS": "0",
                "TOTAL_INPUT_CACHED_TEXT_TOKENS": "0",
                "TOTAL_INPUT_AUDIO_TOKENS": "0",
                "TOTAL_INPUT_CACHED_AUDIO_TOKENS": "0",
                "TOTAL_OUTPUT_TEXT_TOKENS": "0",
                "TOTAL_OUTPUT_AUDIO_TOKENS": "0",
            }
            output_vars.update(token_metrics)

            disconnect_msg = {
                "version": "2",
                "type": "disconnect",
                "seq": self.server_seq + 1,
                "clientseq": self.client_seq,
                "id": self.session_id,
                "parameters": {
                    "reason": reason,
                    "info": info,
                    "outputVariables": output_vars
                }
            }
            self.server_seq += 1

            self.logger.info(f"[FunctionCall] Sending disconnect message to Genesys with {len(output_vars)} output variables")
            await asyncio.wait_for(self._send_json(disconnect_msg), timeout=5.0)

            try:
                await asyncio.wait_for(self.ws.wait_closed(), timeout=5.0)
                self.logger.info(f"[FunctionCall] Genesys acknowledged disconnect for session {self.session_id}")
            except asyncio.TimeoutError:
                self.logger.warning(f"[FunctionCall] Timeout waiting for Genesys to acknowledge disconnect for session {self.session_id}")
        except Exception as e:
            self.logger.error(f"[FunctionCall] Error in disconnect_session: {e}", exc_info=True)
            self.running = False
            await self.stop_audio_processing()
            if self.ai_client:
                try:
                    await self.ai_client.close()
                except Exception:
                    pass

    async def handle_audio_frame(self, frame_bytes: bytes):
        """Forwards one caller audio frame (PCMU 8 kHz) from Genesys to the AI vendor client."""
        if not self.ai_client or not self.ai_client.running or self.rate_limit_state["in_backoff"]:
            return
        self.audio_frames_received += 1
        await self.ai_client.send_audio(frame_bytes)

    async def _send_json(self, msg: dict):
        try:
            if not await self.message_limiter.acquire():
                current_rate = self.message_limiter.get_current_rate()
                self.logger.warning(
                    f"Message rate limit exceeded (current rate: {current_rate:.2f}/s). "
                    f"Message type: {msg.get('type')}. Dropping to maintain compliance."
                )
                return

            self.logger.debug(f"Sending message to Genesys:\n{format_json(msg)}")
            await self.ws.send(json.dumps(msg))
        except websockets.ConnectionClosed:
            self.logger.warning("Genesys WebSocket closed while sending JSON message.")
            self.running = False
