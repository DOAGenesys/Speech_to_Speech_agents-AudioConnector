"""
Offline smoke tests for the OpenAI Realtime, GPT-Live, xAI Grok and Gemini Live clients and the AudioHook server.

Every vendor endpoint is replaced by a local mock (WebSocket servers for OpenAI / xAI, a fake google-genai
session for Gemini, an httpx MockTransport for the Responses API), and every API key is overwritten with an
obviously fake value before the bridge modules are imported, so the tests never reach a real service.

Run from the repository root:  python tests/smoke_test_vendors.py
"""

import asyncio
import base64
import contextlib
import json
import logging
import math
import os
import struct
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

FAKE_KEYS = {
    "GENESYS_API_KEY": "test-genesys-key",
    "OPENAI_API_KEY": "sk-test-xxxxxxxx",
    "GEMINI_API_KEY": "gemini-test-xxxxxxxx",
    "XAI_API_KEY": "xai-test-xxxxxxxx",
}
UNREACHABLE = "127.0.0.1:9"
os.environ.update(FAKE_KEYS)
os.environ.update({
    "AI_VENDOR": "openai",
    "OPENAI_REALTIME_BASE_URL": f"ws://{UNREACHABLE}/v1/realtime",
    "OPENAI_LIVE_URL": f"ws://{UNREACHABLE}/v1/live/sessions",
    "OPENAI_API_BASE_URL": f"http://{UNREACHABLE}/v1",
    "XAI_REALTIME_BASE_URL": f"ws://{UNREACHABLE}/v1/realtime",
    "GEMINI_BASE_URL": UNREACHABLE,
    "GEMINI_THINKING_LEVEL": "low",
    "GENESYS_CLIENT_ID": "",
    "GENESYS_CLIENT_SECRET": "",
})
for _name in ("AI_MODEL", "AI_VOICE", "OPENAI_MODEL", "GEMINI_MODEL", "GROK_MODEL", "GROK_KEYTERMS"):
    os.environ.pop(_name, None)
# config.py writes logging.txt to the working directory.
os.chdir(tempfile.mkdtemp(prefix="audiohook-smoke-"))

import httpx  # noqa: E402
import websockets  # noqa: E402
from websockets.asyncio.server import serve  # noqa: E402
from google.genai import errors as genai_errors, types  # noqa: E402

import config  # noqa: E402
import openai_client  # noqa: E402
import grok_client  # noqa: E402
import gpt_live_client  # noqa: E402
import gemini_client  # noqa: E402
import audio_hook_server  # noqa: E402
from call_control import END_ESCALATION, END_SUCCESS  # noqa: E402

logging.getLogger("GenesysOpenAIBridge").setLevel(logging.WARNING)

LOOKUP_TOOL = {
    "type": "function",
    "name": "lookup_order",
    "description": "Look up an order by id. Returns status.",
    "parameters": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
        "additionalProperties": False,
    },
}
BALANCE_TOOL = {
    "type": "function",
    "name": "check_balance",
    "description": "Check the account balance.",
    "parameters": {"type": "object", "properties": {}},
}
SILENCE_PCMU = b"\xff"
LOUD_PCMU = b"\x00"


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def sine_pcm16(rate: int, seconds: float, freq: float = 440.0, amplitude: int = 8000) -> bytes:
    count = int(rate * seconds)
    return struct.pack(f"<{count}h", *(int(amplitude * math.sin(2 * math.pi * freq * i / rate)) for i in range(count)))


async def eventually(predicate, timeout: float = 5.0, message: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"timed out waiting for {message}")


def tool_handlers(calls):
    async def lookup_order(args):
        calls.append(("lookup_order", args))
        await asyncio.sleep(0.05)
        return {"status": "shipped"}

    async def check_balance(args):
        calls.append(("check_balance", args))
        return {"balance": 12.5}

    return {"lookup_order": lookup_order, "check_balance": check_balance}


class MockRealtimeServer:
    """Local WebSocket server that records client events and replies through a scripted callback."""

    def __init__(self, on_connect=(), auto_reply=None):
        self.on_connect = list(on_connect)
        self.auto_reply = auto_reply
        self.received = []
        self.request_path = None
        self.authorization_matches = None
        self.expected_authorization = None
        self.connections = 0
        self._outbox: asyncio.Queue = asyncio.Queue()
        self._server = None
        self.port = None

    async def __aenter__(self):
        self._server = await serve(self._handler, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        self._server.close()
        await self._server.wait_closed()

    def url(self, path: str) -> str:
        return f"ws://127.0.0.1:{self.port}{path}"

    async def _handler(self, ws):
        self.connections += 1
        self.request_path = ws.request.path
        if self.expected_authorization is not None:
            self.authorization_matches = ws.request.headers.get("Authorization") == self.expected_authorization
        for event in self.on_connect:
            await ws.send(json.dumps(event))

        async def pump():
            while True:
                event = await self._outbox.get()
                await ws.send(json.dumps(event))

        pump_task = asyncio.create_task(pump())
        try:
            async for raw in ws:
                event = json.loads(raw)
                self.received.append(event)
                for reply in (self.auto_reply(event) if self.auto_reply else None) or []:
                    await ws.send(json.dumps(reply))
        except websockets.ConnectionClosed:
            pass
        finally:
            pump_task.cancel()

    def emit(self, *events):
        for event in events:
            self._outbox.put_nowait(event)

    def of_type(self, event_type, predicate=None):
        return [e for e in self.received if e.get("type") == event_type and (predicate is None or predicate(e))]


# ---------------------------------------------------------------------------------------------- OpenAI


async def test_openai_realtime():
    def reply(event):
        if event["type"] == "session.update":
            return [{"type": "session.updated", "session": event["session"]}]
        response = event.get("response") or {}
        if event["type"] == "response.create" and response.get("conversation") == "none":
            return [
                {"type": "response.created", "response": {"id": "resp_sum", "metadata": response.get("metadata")}},
                {"type": "response.output_text.delta", "response_id": "resp_sum", "delta": "Caller checked "},
                {"type": "response.done", "response": {
                    "id": "resp_sum", "status": "completed", "metadata": response.get("metadata"),
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "Caller checked order 42."}]}],
                    "usage": {"input_tokens": 5, "output_tokens": 7,
                              "input_token_details": {"text_tokens": 5}, "output_token_details": {"text_tokens": 7}},
                }},
            ]
        return []

    async with MockRealtimeServer(on_connect=[{"type": "session.created", "session": {}}], auto_reply=reply) as server:
        server.expected_authorization = f"Bearer {FAKE_KEYS['OPENAI_API_KEY']}"
        openai_client.OPENAI_REALTIME_BASE_URL = server.url("/v1/realtime")

        barge_ins, ended, tool_calls, frames = [], [], [], []

        async def on_barge_in():
            barge_ins.append(time.monotonic())

        async def on_end(reason, info):
            ended.append((reason, info))

        client = openai_client.OpenAIRealtimeClient("oai-test", on_speech_started_callback=on_barge_in)
        client.on_end_call_request = on_end
        client.playback_remaining_fn = lambda: 0.0
        client.register_genesys_tool_handlers(tool_handlers(tool_calls))
        await client.connect(instructions="You are a test agent.", voice="marin", model="gpt-realtime-2.1",
                             tool_definitions=[LOOKUP_TOOL, BALANCE_TOOL])

        assert server.authorization_matches, "Authorization header was not the configured key"
        assert server.request_path == "/v1/realtime?model=gpt-realtime-2.1", server.request_path
        session = server.of_type("session.update")[0]["session"]
        assert session["type"] == "realtime"
        assert session["audio"]["input"]["format"] == {"type": "audio/pcmu"}
        assert session["audio"]["output"] == {"format": {"type": "audio/pcmu"}, "voice": "marin"}
        assert session["audio"]["input"]["turn_detection"]["type"] == "semantic_vad"
        assert session["reasoning"] == {"effort": "low"}
        assert [t["name"] for t in session["tools"]] == [END_SUCCESS, END_ESCALATION, "lookup_order", "check_balance"]
        assert "temperature" not in session

        reasoning = {m: openai_client._REASONING_REALTIME_MODEL.match(m) is not None for m in (
            "gpt-realtime-2", "gpt-realtime-2.1", "gpt-realtime-2.1-mini", "gpt-realtime",
            "gpt-realtime-1.5", "gpt-realtime-mini", "gpt-realtime-2025-08-28")}
        assert reasoning == {"gpt-realtime-2": True, "gpt-realtime-2.1": True, "gpt-realtime-2.1-mini": True,
                             "gpt-realtime": False, "gpt-realtime-1.5": False, "gpt-realtime-mini": False,
                             "gpt-realtime-2025-08-28": False}, reasoning

        await client.start_receiving(frames.append)
        await client.send_audio(SILENCE_PCMU * 1600)
        await eventually(lambda: server.of_type("input_audio_buffer.append"), message="caller audio")

        # Turn 1: audio plus two parallel tool calls in one response.
        server.emit(
            {"type": "response.created", "response": {"id": "resp_1"}},
            {"type": "response.output_audio.delta", "response_id": "resp_1", "item_id": "item_1",
             "delta": b64(SILENCE_PCMU * 2000)},
            {"type": "response.output_audio.done", "response_id": "resp_1", "item_id": "item_1"},
            {"type": "response.function_call_arguments.done", "response_id": "resp_1", "call_id": "call_a",
             "name": "lookup_order", "arguments": json.dumps({"order_id": "42"})},
            {"type": "response.function_call_arguments.done", "response_id": "resp_1", "call_id": "call_b",
             "name": "check_balance", "arguments": "{}"},
            {"type": "response.done", "response": {
                "id": "resp_1", "status": "completed",
                "output": [
                    {"type": "function_call", "call_id": "call_a", "name": "lookup_order",
                     "arguments": json.dumps({"order_id": "42"})},
                    {"type": "function_call", "call_id": "call_b", "name": "check_balance", "arguments": "{}"},
                ],
                "usage": {"input_token_details": {"text_tokens": 10, "audio_tokens": 20,
                                                  "cached_tokens_details": {"text_tokens": 1, "audio_tokens": 2}},
                          "output_token_details": {"text_tokens": 3, "audio_tokens": 40}},
            }},
        )
        await eventually(lambda: len(server.of_type("response.create")) == 1, message="response.create after tools")
        await asyncio.sleep(0.2)
        outputs = server.of_type("conversation.item.create", lambda e: e["item"]["type"] == "function_call_output")
        assert sorted(o["item"]["call_id"] for o in outputs) == ["call_a", "call_b"]
        assert len(server.of_type("response.create")) == 1, "expected exactly one response.create for parallel calls"
        last_output = max(server.received.index(o) for o in outputs)
        assert server.received.index(server.of_type("response.create")[0]) > last_output
        assert sorted(name for name, _ in tool_calls) == ["check_balance", "lookup_order"]
        assert [len(f) for f in frames] == [1600, 1600], [len(f) for f in frames]

        # Barge-in while a long answer is still playing truncates the assistant item.
        server.emit(
            {"type": "response.created", "response": {"id": "resp_2"}},
            {"type": "response.output_audio.delta", "response_id": "resp_2", "item_id": "item_2",
             "delta": b64(SILENCE_PCMU * 16000)},
        )
        await eventually(lambda: len(frames) >= 12, message="turn 2 audio")
        server.emit({"type": "input_audio_buffer.speech_started", "audio_start_ms": 1000})
        await eventually(lambda: barge_ins and server.of_type("conversation.item.truncate"), message="truncate")
        truncate = server.of_type("conversation.item.truncate")[0]
        assert truncate["item_id"] == "item_2" and 0 <= truncate["audio_end_ms"] < 2000, truncate

        # Call control: function output, then a farewell response tagged with metadata.
        server.emit({"type": "response.done", "response": {
            "id": "resp_2", "status": "cancelled",
            "output": [{"type": "function_call", "call_id": "call_c", "name": END_SUCCESS,
                        "arguments": json.dumps({"summary": "Order 42 checked"})}],
        }})
        await eventually(lambda: len(server.of_type("response.create")) == 2, message="farewell response.create")
        farewell = server.of_type("response.create")[1]["response"]
        assert farewell["metadata"] == {"type": "final_farewell"} and farewell["tool_choice"] == "none"
        assert not ended, "call ended before the farewell was spoken"
        server.emit(
            {"type": "response.created", "response": {"id": "resp_3", "metadata": {"type": "final_farewell"}}},
            {"type": "response.output_audio.delta", "response_id": "resp_3", "item_id": "item_3",
             "delta": b64(SILENCE_PCMU * 800)},
            {"type": "response.done", "response": {"id": "resp_3", "status": "completed", "output": []}},
        )
        await eventually(lambda: ended, message="end-call callback")
        assert ended == [("completed", "Order 42 checked")], ended

        summary = await client.generate_summary("Summarize the call.", timeout=3)
        assert summary == "Caller checked order 42.", summary
        metrics = client.get_token_metrics()
        assert metrics["TOTAL_INPUT_TEXT_TOKENS"] == "15" and metrics["TOTAL_INPUT_AUDIO_TOKENS"] == "20"
        assert metrics["TOTAL_INPUT_CACHED_TEXT_TOKENS"] == "1" and metrics["TOTAL_OUTPUT_AUDIO_TOKENS"] == "40"
        await client.close()
        assert not client.running


# ---------------------------------------------------------------------------------------------- Grok


def grok_reply(summary_text="Resumen de la llamada."):
    def reply(event):
        if event["type"] == "session.update":
            return [{"type": "session.updated", "session": event["session"]}]
        item = event.get("item") or {}
        if event["type"] == "conversation.item.create" and item.get("type") == "force_message":
            return [
                {"type": "response.created", "response": {"id": "resp_fm", "metadata": None}},
                {"type": "response.output_audio.delta", "response_id": "resp_fm", "item_id": "item_fm",
                 "delta": b64(SILENCE_PCMU * 1600)},
                {"type": "response.output_audio.done", "response_id": "resp_fm", "item_id": "item_fm"},
                {"type": "response.done", "response": {"id": "resp_fm", "status": "completed", "metadata": None,
                                                       "output": []}},
            ]
        response = event.get("response") or {}
        if event["type"] == "response.create" and response.get("modalities") == ["text"]:
            return [
                {"type": "response.created", "response": {"id": "resp_sum", "metadata": None}},
                {"type": "response.output_text.delta", "response_id": "resp_sum", "delta": summary_text},
                {"type": "response.done", "response": {
                    "id": "resp_sum", "status": "completed", "metadata": None,
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": summary_text}]}],
                    "usage": {"input_tokens": 30, "output_tokens": 8, "total_tokens": 38},
                }},
            ]
        return []

    return reply


async def test_grok_voice():
    async with MockRealtimeServer(auto_reply=grok_reply()) as server:
        server.expected_authorization = f"Bearer {FAKE_KEYS['XAI_API_KEY']}"
        grok_client.XAI_REALTIME_BASE_URL = server.url("/v1/realtime")

        ended, frames = [], []

        async def on_end(reason, info):
            ended.append((reason, info))

        client = grok_client.GrokRealtimeClient("grok-test")
        client.language = "es-ES"
        client.success_prompt = "Gracias por llamar, adiós."
        client.on_end_call_request = on_end
        client.playback_remaining_fn = lambda: 0.0
        client.set_keyterms("Genesys, AudioHook,   ")
        nested_tool = {"type": "function", "function": {**{k: v for k, v in LOOKUP_TOOL.items() if k != "type"},
                                                        "parameters": {**LOOKUP_TOOL["parameters"], "strict": True}}}
        await client.connect(instructions="Eres un agente de prueba.", voice="eve", model="grok-voice-3.0",
                             tool_definitions=[nested_tool, {"type": "web_search_preview", "search_context_size": "low"},
                                               {"type": "code_interpreter"}])

        assert server.authorization_matches
        assert server.request_path == "/v1/realtime?model=grok-voice-3.0", server.request_path
        assert not server.of_type("session.created")
        session = server.of_type("session.update")[0]["session"]
        assert "type" not in session and session["voice"] == "eve"
        assert session["turn_detection"] == {"type": "server_vad"}
        assert session["audio"]["input"]["format"] == {"type": "audio/pcmu"}
        assert session["audio"]["output"]["format"] == {"type": "audio/pcmu"}
        assert session["audio"]["input"]["transcription"] == {"language_hint": "es-ES", "keyterms": ["Genesys", "AudioHook"]}
        tools = session["tools"]
        assert [t.get("name", t["type"]) for t in tools] == [END_SUCCESS, END_ESCALATION, "lookup_order", "web_search"]
        assert "strict" not in tools[2]["parameters"] and "function" not in tools[2]
        assert tools[3] == {"type": "web_search"}

        await client.start_receiving(frames.append)
        server.emit(
            {"type": "response.created", "response": {"id": "resp_1", "metadata": None}},
            {"type": "response.output_item.done", "response_id": "resp_1", "item": {
                "type": "function_call", "call_id": "call_x", "name": END_SUCCESS,
                "arguments": json.dumps({"summary": "Pedido revisado"})}},
            {"type": "response.done", "response": {"id": "resp_1", "status": "completed", "metadata": None,
                                                   "usage": {"input_tokens": 100, "output_tokens": 50,
                                                             "total_tokens": 150}}},
        )
        await eventually(lambda: ended, message="Grok end-call callback")
        assert ended == [("completed", "Pedido revisado")], ended
        force = server.of_type("conversation.item.create", lambda e: e["item"]["type"] == "force_message")
        assert force and force[0]["item"]["content"] == [{"type": "output_text", "text": "Gracias por llamar, adiós."}]
        assert not server.of_type("response.create"), "force_message must not be followed by response.create"
        assert sum(len(f) for f in frames) == 1600

        summary = await client.generate_summary("Resume la llamada.", timeout=3)
        assert summary == "Resumen de la llamada.", summary
        metrics = client.get_token_metrics()
        assert metrics["TOTAL_INPUT_AUDIO_TOKENS"] == "130" and metrics["TOTAL_OUTPUT_AUDIO_TOKENS"] == "58", metrics
        await client.close()


# ---------------------------------------------------------------------------------------------- GPT-Live


async def test_gpt_live():
    def reply(event):
        if event["type"] == "session.start":
            return [{"type": "session.started", "session": {"id": "sess_live_1", **event["session"]}}]
        if event["type"] == "session.close":
            return [{"type": "session.closed", "reason": "client", "usage": {"seconds": 42.5}}]
        return []

    async with MockRealtimeServer(auto_reply=reply) as server:
        server.expected_authorization = f"Bearer {FAKE_KEYS['OPENAI_API_KEY']}"
        gpt_live_client.OPENAI_LIVE_URL = server.url("/v1/live/sessions")
        gpt_live_client.OPENAI_API_BASE_URL = "http://responses.mock/v1"

        summary_requests = []

        def responses_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            summary_requests.append({
                "path": request.url.path,
                "auth_ok": request.headers.get("Authorization") == f"Bearer {FAKE_KEYS['OPENAI_API_KEY']}",
                "body": body,
            })
            return httpx.Response(200, json={"output_text": "Caller asked for a human.",
                                             "usage": {"input_tokens": 50, "output_tokens": 9}})

        real_async_client = httpx.AsyncClient
        gpt_live_client.httpx.AsyncClient = lambda **kwargs: real_async_client(
            transport=httpx.MockTransport(responses_handler), **kwargs)

        handoffs, ended, tool_calls, frames = [], [], [], []

        async def on_handoff(reason, info):
            handoffs.append((reason, info))

        async def on_end(reason, info):
            ended.append((reason, info))

        client = gpt_live_client.GptLiveClient("live-test")
        client.on_handoff_request = on_handoff
        client.on_end_call_request = on_end
        client.register_genesys_tool_handlers(tool_handlers(tool_calls))
        try:
            await client.connect(instructions="You are a test agent.", voice="marin", model="gpt-live-1",
                                 tool_definitions=[LOOKUP_TOOL, BALANCE_TOOL])

            assert server.authorization_matches and server.request_path == "/v1/live/sessions"
            start = server.of_type("session.start")[0]["session"]
            assert start["model"] == "gpt-live-1"
            assert start["audio"] == {"format": {"type": "audio/pcmu", "rate": 8000}, "output": {"voice": "marin"}}
            delegation = start["delegation"]
            assert delegation["type"] == "responses" and delegation["responses"]["model"] == "gpt-6-luna"
            strict = {t["name"]: t.get("strict") for t in delegation["responses"]["tools"]}
            assert strict == {END_SUCCESS: True, END_ESCALATION: True, "lookup_order": False, "check_balance": False}
            assert "Delegation policy" in start["instructions"] and "[CALL CONTROL]" not in start["instructions"]
            assert client.live_session_id == "sess_live_1"

            await client.start_receiving(frames.append)
            await client.send_audio(SILENCE_PCMU * 1600)
            await eventually(lambda: server.of_type("session.input_audio.append"), message="live caller audio")

            server.emit(
                {"type": "session.input_transcript.delta", "delta": "Where is order 42?"},
                {"type": "session.output_audio.delta", "delta": b64(SILENCE_PCMU * 1000)},
                {"type": "session.output_transcript.delta", "delta": "Let me check."},
                {"type": "session.delegation.created", "delegation": {"id": "del_1", "target": "responses"}},
                {"type": "response.event", "delegation_id": "del_1", "event": {
                    "type": "response.output_item.done", "item": {
                        "type": "function_call", "call_id": "c1", "name": "lookup_order",
                        "arguments": json.dumps({"order_id": "42"})}}},
                {"type": "response.event", "delegation_id": "del_1", "event": {
                    "type": "response.output_item.done", "item": {
                        "type": "function_call", "call_id": "c2", "name": "check_balance", "arguments": "{}"}}},
                {"type": "response.event", "delegation_id": "del_1", "event": {
                    "type": "response.completed", "response": {
                        "usage": {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 40},
                                  "output_tokens": 20}}}},
                {"type": "session.usage.updated", "usage": {"seconds": 10.0}},
            )
            await eventually(lambda: len(server.of_type("response.create")) == 1, message="live continuation")
            outputs = server.of_type("response.item.create", lambda e: e["item"]["type"] == "function_call_output")
            assert sorted(o["item"]["call_id"] for o in outputs) == ["c1", "c2"]
            await eventually(lambda: frames, timeout=2, message="idle flush of live audio")
            assert [len(f) for f in frames] == [1600], [len(f) for f in frames]

            server.emit(
                {"type": "response.event", "delegation_id": "del_2", "event": {
                    "type": "response.output_item.done", "item": {
                        "type": "function_call", "call_id": "c3", "name": END_ESCALATION,
                        "arguments": json.dumps({"reason": "Caller wants a human"})}}},
                {"type": "response.event", "delegation_id": "del_2", "event": {"type": "response.completed",
                                                                               "response": {}}},
            )
            await eventually(lambda: server.of_type("session.instructions.append"), message="farewell instruction")
            append = server.of_type("session.instructions.append")[0]
            assert append["delegation_id"] is None and "live agent" in append["content"]
            assert len(server.of_type("response.create")) == 1, "call control must not trigger response.create"
            await asyncio.sleep(0.2)
            assert not handoffs, "handoff fired before the farewell audio"
            server.emit({"type": "session.output_audio.delta", "delta": b64(LOUD_PCMU * 1600)})
            await eventually(lambda: handoffs, timeout=4, message="handoff after farewell")
            assert handoffs == [("transfer", "Caller wants a human")] and not ended

            summary = await client.generate_summary("Summarize the call.", timeout=3)
            assert summary == "Caller asked for a human.", summary
            request = summary_requests[0]
            assert request["path"] == "/v1/responses" and request["auth_ok"]
            assert request["body"]["model"] == "gpt-6-luna" and request["body"]["store"] is False
            assert "Caller: Where is order 42?" in request["body"]["input"]

            await client.close()
            assert server.of_type("session.close")
            metrics = client.get_token_metrics()
            assert metrics["TOTAL_VOICE_SECONDS"] == "42.5", metrics
            assert metrics["TOTAL_INPUT_TEXT_TOKENS"] == "110" and metrics["TOTAL_INPUT_CACHED_TEXT_TOKENS"] == "40"
            assert metrics["TOTAL_OUTPUT_TEXT_TOKENS"] == "29", metrics
        finally:
            gpt_live_client.httpx.AsyncClient = real_async_client


# ---------------------------------------------------------------------------------------------- Gemini

_SESSION_CLOSED = object()


class FakeLiveSession:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.audio = []
        self.tool_responses = []
        self.closed = False

    async def send_realtime_input(self, audio=None, **kwargs):
        self.audio.append(audio)

    async def send_tool_response(self, function_responses):
        self.tool_responses.append(list(function_responses))

    async def receive(self):
        while True:
            message = await self.inbox.get()
            if message is _SESSION_CLOSED:
                raise genai_errors.APIError(1000, {"message": "session closed"}, None)
            yield message
            content = message.server_content
            if content is not None and content.turn_complete:
                return

    def push(self, **fields):
        self.inbox.put_nowait(types.LiveServerMessage(**fields))


class FakeGenaiClient:
    def __init__(self):
        self.sessions = []
        self.summary_calls = []
        self.aio = SimpleNamespace(live=SimpleNamespace(connect=self._connect),
                                   models=SimpleNamespace(generate_content=self._generate_content))

    @contextlib.asynccontextmanager
    async def _connect(self, model, config):
        session = FakeLiveSession(model, config)
        self.sessions.append(session)
        try:
            yield session
        finally:
            session.closed = True
            session.inbox.put_nowait(_SESSION_CLOSED)

    async def _generate_content(self, model, contents, config):
        self.summary_calls.append({"model": model, "contents": contents, "system": config.system_instruction})
        return SimpleNamespace(text="Gemini summary.",
                               usage_metadata=SimpleNamespace(prompt_token_count=7, candidates_token_count=5))


def audio_message(pcm16_24k: bytes):
    return {"server_content": types.LiveServerContent(model_turn=types.Content(parts=[
        types.Part(inline_data=types.Blob(data=pcm16_24k, mime_type="audio/pcm;rate=24000"))]))}


async def test_gemini_live():
    fake = FakeGenaiClient()
    original_factory = gemini_client.GeminiRealtimeClient._create_sdk_client
    gemini_client.GeminiRealtimeClient._create_sdk_client = lambda self: fake
    try:
        barge_ins, handoffs, ended, tool_calls, frames = [], [], [], [], []

        async def on_barge_in():
            barge_ins.append(1)

        async def on_handoff(reason, info):
            handoffs.append((reason, info))

        async def on_end(reason, info):
            ended.append((reason, info))

        client = gemini_client.GeminiRealtimeClient("gem-test", on_speech_started_callback=on_barge_in)
        client.on_handoff_request = on_handoff
        client.on_end_call_request = on_end
        client.playback_remaining_fn = lambda: 0.0
        client.register_genesys_tool_handlers(tool_handlers(tool_calls))
        await client.connect(instructions="You are a test agent.", voice="Kore", model="gemini-3.8-live",
                             tool_definitions=[LOOKUP_TOOL])

        session = fake.sessions[0]
        cfg = session.config
        assert session.model == "gemini-3.8-live"
        assert cfg.response_modalities == [types.Modality.AUDIO]
        assert cfg.speech_config.voice_config.prebuilt_voice_config.voice_name == "Kore"
        assert cfg.thinking_config is None, "gemini-3.8-live must not receive thinking_config"
        assert cfg.temperature is None and cfg.max_output_tokens is None
        assert cfg.session_resumption is not None and cfg.context_window_compression.sliding_window is not None
        vad = cfg.realtime_input_config.automatic_activity_detection
        assert vad.silence_duration_ms == 700 and vad.prefix_padding_ms == 100
        declarations = {d.name: d for d in cfg.tools[0].function_declarations}
        assert declarations[END_SUCCESS].behavior == types.Behavior.BLOCKING
        assert declarations["lookup_order"].behavior is None
        assert "additionalProperties" not in declarations["lookup_order"].parameters_json_schema

        profile_checks = {}
        for model in ("gemini-3.8-live-extended-thinking", "gemini-3.1-flash-live-preview"):
            probe = gemini_client.GeminiRealtimeClient("probe")
            probe.model = model
            probe.profile = gemini_client.gemini_live_profile(model)
            probe_cfg = probe._build_live_config("prompt", [LOOKUP_TOOL], None, None)
            profile_checks[model] = (
                probe_cfg.thinking_config.thinking_level if probe_cfg.thinking_config else None,
                {d.behavior for d in probe_cfg.tools[0].function_declarations},
            )
        assert profile_checks["gemini-3.8-live-extended-thinking"] == (types.ThinkingLevel.LOW, {types.Behavior.NON_BLOCKING})
        assert profile_checks["gemini-3.1-flash-live-preview"] == (types.ThinkingLevel.LOW, {None})

        await client.start_receiving(frames.append)
        await client.send_audio(SILENCE_PCMU * 1600)
        assert session.audio[0].mime_type == "audio/pcm;rate=8000" and len(session.audio[0].data) == 3200

        # One model turn (0.5 s of 24 kHz audio), transcripts and usage, across a turn boundary.
        session.push(session_resumption_update=types.LiveServerSessionResumptionUpdate(new_handle="handle-1",
                                                                                      resumable=True))
        session.push(server_content=types.LiveServerContent(input_transcription=types.Transcription(text="Hi there")))
        session.push(**audio_message(sine_pcm16(24000, 0.5)))
        session.push(server_content=types.LiveServerContent(output_transcription=types.Transcription(text="Hello!")))
        session.push(server_content=types.LiveServerContent(turn_complete=True),
                     usage_metadata=types.UsageMetadata(
                         prompt_tokens_details=[types.ModalityTokenCount(modality=types.MediaModality.AUDIO, token_count=25),
                                                types.ModalityTokenCount(modality=types.MediaModality.TEXT, token_count=300)],
                         response_tokens_details=[types.ModalityTokenCount(modality=types.MediaModality.AUDIO,
                                                                           token_count=60)]))
        await eventually(lambda: len(frames) == 3, message="Gemini audio frames")
        assert all(len(f) == 1600 for f in frames)
        assert client._resumption_handle == "handle-1"

        # The receive loop must survive the end of the first turn and run tools in the next one.
        session.push(tool_call=types.LiveServerToolCall(function_calls=[
            types.FunctionCall(id="fc1", name="lookup_order", args={"order_id": "42"})]))
        await eventually(lambda: session.tool_responses, message="Gemini tool response")
        response = session.tool_responses[0][0]
        assert response.id == "fc1" and response.response["status"] == "ok"
        assert response.response["scheduling"] == "WHEN_IDLE"

        session.push(server_content=types.LiveServerContent(interrupted=True))
        await eventually(lambda: barge_ins, message="Gemini barge-in callback")

        # GoAway: resume on a new connection with the latest handle.
        session.push(go_away=types.LiveServerGoAway(time_left="5s"))
        await eventually(lambda: len(fake.sessions) == 2, message="Gemini resumption")
        resumed = fake.sessions[1]
        assert resumed.config.session_resumption.handle == "handle-1"
        await eventually(lambda: session.closed, message="old Gemini session closed")
        assert client.running

        resumed.push(tool_call=types.LiveServerToolCall(function_calls=[
            types.FunctionCall(id="fc2", name=END_ESCALATION, args={"reason": "Caller wants a human"})]))
        await eventually(lambda: resumed.tool_responses, message="Gemini call-control response")
        control = resumed.tool_responses[0][0]
        assert control.response["action"] == END_ESCALATION and "instruction" in control.response
        assert "scheduling" not in control.response, "BLOCKING declarations take no scheduling"
        assert not handoffs
        resumed.push(**audio_message(sine_pcm16(24000, 0.3)))
        resumed.push(server_content=types.LiveServerContent(turn_complete=True))
        await eventually(lambda: handoffs, message="Gemini handoff after farewell")
        assert handoffs == [("transfer", "Caller wants a human")] and not ended

        summary = await client.generate_summary("Summarize the call.", timeout=3)
        assert summary == "Gemini summary."
        call = fake.summary_calls[0]
        assert call["model"] == config.GEMINI_SUMMARY_MODEL and "Caller: Hi there" in call["contents"]
        metrics = client.get_token_metrics()
        assert metrics["TOTAL_INPUT_AUDIO_TOKENS"] == "25" and metrics["TOTAL_INPUT_TEXT_TOKENS"] == "307", metrics
        assert metrics["TOTAL_OUTPUT_AUDIO_TOKENS"] == "60" and metrics["TOTAL_OUTPUT_TEXT_TOKENS"] == "5", metrics
        await client.close()
        assert resumed.closed and not client.running
    finally:
        gemini_client.GeminiRealtimeClient._create_sdk_client = original_factory


# ---------------------------------------------------------------------------------------------- AudioHook


class FakeGenesysSocket:
    def __init__(self):
        self.json_messages = []
        self.audio_frames = []

    async def send(self, data):
        if isinstance(data, (bytes, bytearray)):
            self.audio_frames.append(bytes(data))
        else:
            self.json_messages.append(json.loads(data))

    async def wait_closed(self):
        return None

    def of_type(self, message_type):
        return [m for m in self.json_messages if m["type"] == message_type]


def genesys_open(input_vars):
    return {
        "version": "2", "type": "open", "seq": 1, "id": "e2e-session",
        "parameters": {
            "organizationId": "org", "conversationId": "conv-1",
            "participant": {"id": "participant-1", "ani": "", "dnis": ""},
            "media": [{"type": "audio", "format": "PCMU", "channels": ["external"], "rate": 8000}],
            "inputVariables": input_vars,
        },
    }


async def test_audiohook_grok_end_to_end():
    async with MockRealtimeServer(auto_reply=grok_reply("Test summary.")) as server:
        grok_client.XAI_REALTIME_BASE_URL = server.url("/v1/realtime")
        genesys = FakeGenesysSocket()
        hook = audio_hook_server.AudioHookServer(genesys)
        await hook.handle_message(genesys_open({
            "AI_VENDOR": "xai",
            "OPENAI_MODEL": "gpt-realtime-2.1",
            "AI_SYSTEM_PROMPT": "You are a test agent.",
            "LANGUAGE": "en-US",
            "SUCCESS_PROMPT": "Thanks for calling, goodbye.",
            "GROK_KEYTERMS": "Genesys",
            "GROK_REASONING_EFFORT": "none",
        }))
        assert genesys.of_type("opened"), genesys.json_messages
        assert isinstance(hook.ai_client, grok_client.GrokRealtimeClient)
        assert server.request_path == "/v1/realtime?model=grok-voice-3.0", server.request_path
        session = server.of_type("session.update")[0]["session"]
        assert session["audio"]["input"]["transcription"] == {"language_hint": "en-US", "keyterms": ["Genesys"]}
        assert session["reasoning"] == {"effort": "none"}

        await hook.handle_audio_frame(SILENCE_PCMU * 1600)
        await eventually(lambda: server.of_type("input_audio_buffer.append"), message="caller audio to Grok")

        server.emit(
            {"type": "response.created", "response": {"id": "r1", "metadata": None}},
            {"type": "response.output_audio.delta", "response_id": "r1", "item_id": "i1",
             "delta": b64(SILENCE_PCMU * 3200)},
        )
        await eventually(lambda: len(genesys.audio_frames) == 2, message="audio frames to Genesys")
        server.emit({"type": "input_audio_buffer.speech_started"})
        await eventually(lambda: genesys.of_type("event"), message="barge_in event")
        assert genesys.of_type("event")[0]["parameters"]["entities"][0]["type"] == "barge_in"

        server.emit(
            {"type": "response.output_item.done", "response_id": "r1", "item": {
                "type": "function_call", "call_id": "cx", "name": END_SUCCESS,
                "arguments": json.dumps({"summary": "Done"})}},
            {"type": "response.done", "response": {"id": "r1", "status": "completed", "metadata": None,
                                                   "usage": {"input_tokens": 11, "output_tokens": 22}}},
        )
        await eventually(lambda: genesys.of_type("disconnect"), timeout=15, message="Genesys disconnect")
        disconnect = genesys.of_type("disconnect")[0]["parameters"]
        assert disconnect["reason"] == "completed" and disconnect["info"] == "Done", disconnect
        output = disconnect["outputVariables"]
        assert output["CONVERSATION_SUMMARY"] == "Test summary." and output["ESCALATION_REQUIRED"] == "false"
        assert output["COMPLETION_SUMMARY"] == "Done", output
        assert output["TOTAL_INPUT_AUDIO_TOKENS"] == "41" and output["TOTAL_OUTPUT_AUDIO_TOKENS"] == "30", output
        assert not hook.ai_client.running


async def test_audiohook_vendor_selection():
    hook = audio_hook_server.AudioHookServer(FakeGenesysSocket())
    assert hook._select_model("grok", {"OPENAI_MODEL": "gpt-realtime-2.1"}) == config.GROK_MODEL
    assert hook._select_model("gemini", {"GEMINI_MODEL": "gemini-3.1-flash-live-preview"}) == "gemini-3.1-flash-live-preview"
    assert hook._select_model("openai", {"OPENAI_MODEL": "gpt-live-1", "AI_MODEL": "gpt-realtime-2.1-mini"}) == "gpt-live-1"
    assert hook._select_voice("gemini", {"AI_VOICE": "puck"}) == "Puck"
    assert hook._select_voice("gemini", {"GEMINI_VOICE": "alloy"}) == config.VENDOR_DEFAULT_VOICES["gemini"]
    assert isinstance(hook._create_ai_client("openai", "gpt-live-1"), gpt_live_client.GptLiveClient)
    assert type(hook._create_ai_client("openai", "gpt-realtime-2.1")) is openai_client.OpenAIRealtimeClient

    original = config.XAI_API_KEY
    config.XAI_API_KEY = None
    try:
        genesys = FakeGenesysSocket()
        hook = audio_hook_server.AudioHookServer(genesys)
        await hook.handle_message(genesys_open({"AI_VENDOR": "grok"}))
        assert not genesys.of_type("opened")
        info = genesys.of_type("disconnect")[0]["parameters"]["info"]
        assert "XAI_API_KEY" in info and FAKE_KEYS["XAI_API_KEY"] not in info
    finally:
        config.XAI_API_KEY = original


async def test_middleware_handshake_and_close():
    import oai_middleware

    def openai_reply(event):
        if event["type"] == "session.update":
            return [{"type": "session.updated", "session": event["session"]}]
        response = event.get("response") or {}
        if event["type"] == "response.create" and response.get("conversation") == "none":
            return [
                {"type": "response.created", "response": {"id": "rs", "metadata": response.get("metadata")}},
                {"type": "response.done", "response": {"id": "rs", "status": "completed",
                                                       "metadata": response.get("metadata"),
                                                       "output": [{"type": "message", "content": [
                                                           {"type": "output_text", "text": "Short call."}]}]}},
            ]
        return []

    async with MockRealtimeServer(on_connect=[{"type": "session.created", "session": {}}],
                                  auto_reply=openai_reply) as vendor:
        openai_client.OPENAI_REALTIME_BASE_URL = vendor.url("/v1/realtime")
        server = await serve(oai_middleware.handle_genesys_connection, "127.0.0.1", 0,
                             process_request=oai_middleware.validate_request, max_size=64000,
                             ping_interval=None, ping_timeout=None)
        port = server.sockets[0].getsockname()[1]
        url = f"ws://127.0.0.1:{port}{config.GENESYS_PATH}"
        audiohook_headers = {
            "audiohook-organization-id": "org-1",
            "audiohook-correlation-id": "corr-1",
            "audiohook-session-id": "e2e-session",
        }
        try:
            async with httpx.AsyncClient() as http:
                health = await http.get(f"http://127.0.0.1:{port}/")
            assert health.status_code == 200, health.status_code

            try:
                async with websockets.connect(url, additional_headers={**audiohook_headers, "x-api-key": "wrong-key"}):
                    raise AssertionError("connection with a wrong x-api-key was accepted")
            except websockets.InvalidStatus as exc:
                assert exc.response.status_code == 401, exc.response.status_code

            headers = {**audiohook_headers, "x-api-key": FAKE_KEYS["GENESYS_API_KEY"]}
            async with websockets.connect(url, additional_headers=headers) as genesys:
                await genesys.send(json.dumps(genesys_open({"AI_SYSTEM_PROMPT": "You are a test agent."})))
                opened = json.loads(await asyncio.wait_for(genesys.recv(), timeout=5))
                assert opened["type"] == "opened" and opened["parameters"]["media"][0]["format"] == "PCMU"
                await eventually(lambda: vendor.of_type("session.update"), message="OpenAI session configured")
                await asyncio.sleep(0.1)
                await genesys.send(SILENCE_PCMU * 1600)
                await eventually(lambda: vendor.of_type("input_audio_buffer.append"), message="audio relayed")

                await genesys.send(json.dumps({"version": "2", "type": "ping", "seq": 2, "id": "e2e-session",
                                               "parameters": {}}))
                pong = json.loads(await asyncio.wait_for(genesys.recv(), timeout=5))
                assert pong["type"] == "pong" and pong["clientseq"] == 2

                await genesys.send(json.dumps({"version": "2", "type": "close", "seq": 3, "id": "e2e-session",
                                               "parameters": {"reason": "end"}}))
                closed = json.loads(await asyncio.wait_for(genesys.recv(), timeout=15))
                assert closed["type"] == "closed" and closed["parameters"]["summary"] == "Short call.", closed
        finally:
            server.close()
            await server.wait_closed()


TESTS = [
    test_openai_realtime,
    test_grok_voice,
    test_gpt_live,
    test_gemini_live,
    test_audiohook_grok_end_to_end,
    test_audiohook_vendor_selection,
    test_middleware_handshake_and_close,
]


async def main() -> int:
    failures = 0
    for test in TESTS:
        started = time.monotonic()
        try:
            await asyncio.wait_for(test(), timeout=60)
        except Exception:
            failures += 1
            print(f"FAIL {test.__name__} ({time.monotonic() - started:.1f}s)")
            traceback.print_exc()
        else:
            print(f"PASS {test.__name__} ({time.monotonic() - started:.1f}s)")
    print(f"{len(TESTS) - failures}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
