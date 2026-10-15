"""
Behaviour shared by every speech-to-speech vendor client: Genesys data action execution, call-ending
farewell handling, playback-aware waits, token metrics and background task bookkeeping.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from config import logger
from call_control import CallControlResolution
from utils import TokenUsage

FAREWELL_TIMEOUT_SECONDS = 20.0
MAX_PLAYBACK_WAIT_SECONDS = 8.0
PLAYBACK_WAIT_SLACK_SECONDS = 0.3

ToolHandler = Callable[[Dict[str, Any]], Awaitable[Any]]


@dataclass
class PendingFunctionCall:
    call_id: str
    name: str
    arguments: Any

    def parsed_arguments(self) -> Dict[str, Any]:
        """Return the arguments as a dict; raises ValueError when the model sent invalid JSON."""
        if isinstance(self.arguments, dict):
            return self.arguments
        if not self.arguments:
            return {}
        value = json.loads(self.arguments)
        if not isinstance(value, dict):
            raise ValueError("Function arguments must be a JSON object")
        return value


class BaseVoiceClient:
    vendor = ""
    vendor_label = "Voice"
    api_key_env_name = ""
    default_model = ""
    default_voice = ""

    def __init__(self, session_id: str, on_speech_started_callback=None, api_key: Optional[str] = None):
        self.session_id = session_id
        self.api_key = api_key or self._default_api_key()
        self.logger = logger.getChild(f"{self.vendor_label}Client_{session_id}")
        self.running = False
        self.start_time = time.time()
        self._closing = False
        self._background_tasks = set()

        self.on_speech_started_callback = on_speech_started_callback
        self.on_end_call_request = None
        self.on_handoff_request = None
        self.on_connection_lost = None
        self.playback_remaining_fn: Optional[Callable[[], float]] = None

        self.language: Optional[str] = None
        self.customer_data: Optional[str] = None
        self.escalation_prompt: Optional[str] = None
        self.success_prompt: Optional[str] = None

        self.model = self.default_model
        self.voice = self.default_voice
        self.genesys_tool_handlers: Dict[str, ToolHandler] = {}
        self.usage = TokenUsage()

        self._summary_future: Optional[asyncio.Future] = None
        self._farewell: Optional[CallControlResolution] = None
        self._farewell_timeout_task: Optional[asyncio.Task] = None
        self._disconnect_triggered = False

    def _default_api_key(self) -> Optional[str]:
        return None

    # ------------------------------------------------------------------ tools

    def register_genesys_tool_handlers(self, handlers: Optional[Dict[str, ToolHandler]]):
        self.genesys_tool_handlers = dict(handlers or {})

    async def _run_tool(self, call: PendingFunctionCall) -> Dict[str, Any]:
        handler = self.genesys_tool_handlers.get(call.name)
        if handler is None:
            self.logger.warning(f"Model called unknown function {call.name}")
            return {"status": "error", "tool": call.name, "error_type": "unknown_function",
                    "message": f"Unknown function: {call.name}"}
        try:
            args = call.parsed_arguments()
        except ValueError as exc:
            return {"status": "error", "tool": call.name, "error_type": "invalid_arguments", "message": str(exc)}
        started = time.monotonic()
        try:
            result = await handler(args)
        except Exception as exc:
            self.logger.error(f"Tool {call.name} failed after {time.monotonic() - started:.2f}s: {type(exc).__name__}")
            return {"status": "error", "tool": call.name, "error_type": type(exc).__name__, "message": str(exc)}
        self.logger.info(f"Tool {call.name} completed in {time.monotonic() - started:.2f}s (args: {sorted(args)})")
        return {"status": "ok", "tool": call.name, "result": result if result is not None else {}}

    # ------------------------------------------------------------------ call ending

    async def _wait_for_playback(self, max_wait: float = MAX_PLAYBACK_WAIT_SECONDS):
        remaining_fn = self.playback_remaining_fn
        if remaining_fn is None:
            return
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            try:
                if remaining_fn() <= PLAYBACK_WAIT_SLACK_SECONDS:
                    return
            except Exception:
                return
            await asyncio.sleep(0.1)

    async def _begin_farewell(self, resolution: CallControlResolution, wait_for_playback: bool = True):
        self._farewell = resolution
        self.logger.info(f"Call control requested: {resolution.action}")
        if wait_for_playback:
            await self._wait_for_playback()
        await self._send_farewell(resolution)
        self._farewell_timeout_task = self._spawn(self._farewell_timeout())

    async def _send_farewell(self, resolution: CallControlResolution):
        raise NotImplementedError

    async def _farewell_timeout(self):
        await asyncio.sleep(FAREWELL_TIMEOUT_SECONDS)
        await self._finish_farewell("farewell timeout")

    def _cancel_farewell_timeout(self):
        task = self._farewell_timeout_task
        if task and task is not asyncio.current_task() and not task.done():
            task.cancel()
        self._farewell_timeout_task = None

    async def _finish_farewell(self, reason: str):
        if self._disconnect_triggered or self._farewell is None:
            return
        self._disconnect_triggered = True
        self._cancel_farewell_timeout()
        resolution = self._farewell
        self.logger.info(f"Ending call ({reason}): {resolution.action}")
        callback = (self.on_handoff_request or self.on_end_call_request) if resolution.is_escalation else self.on_end_call_request
        if callback:
            await callback(resolution.disconnect_reason, resolution.info)

    def _notify_connection_lost(self):
        if not self._closing and self.on_connection_lost:
            self._spawn(self.on_connection_lost())

    # ------------------------------------------------------------------ summary & metrics

    def _resolve_summary(self, text: Optional[str]):
        future = self._summary_future
        if future is not None and not future.done():
            future.set_result(text)

    def get_token_metrics(self) -> Dict[str, str]:
        return self.usage.as_output_variables()

    # ------------------------------------------------------------------ helpers

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._on_background_task_done)
        return task

    def _on_background_task_done(self, task: asyncio.Task):
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self.logger.error(f"Background task failed: {type(exc).__name__}: {exc}")

    def _cancel_background_tasks(self):
        current = asyncio.current_task()
        for task in list(self._background_tasks):
            if task is not current and not task.done():
                task.cancel()
