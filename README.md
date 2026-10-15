# Genesys Cloud Real-Time Voice AI Connector

Connect Genesys Cloud phone calls to a speech-to-speech AI voice agent. The connector is a small Python WebSocket server: Genesys Cloud streams the caller's audio to it through the **Audio Connector** (AudioHook), the connector relays it to the AI model you choose, and plays the model's spoken answer back to the caller.

Supported AI vendors:

| Vendor | Default model | Other models you can use |
|--------|---------------|--------------------------|
| **OpenAI Realtime** | `gpt-realtime-2.1` | `gpt-realtime-2.1-mini`, `gpt-realtime-2`, `gpt-realtime` |
| **OpenAI GPT-Live** | `gpt-live-1` | (tools run on `gpt-6-luna`, configurable) |
| **Google Gemini Live** | `gemini-3.8-live` | `gemini-3.8-live-extended-thinking`, `gemini-3.1-flash-live-preview` |
| **xAI Grok Voice** | `grok-voice-3.0` | `grok-voice-latest` |

You pick a default vendor for the server, and each call can switch vendor from the Architect flow.

**What the voice agent can do**

- Talk naturally with the caller, including interruptions (barge-in).
- Look up and update data by calling your **Genesys Cloud Data Actions** (and, optionally, remote MCP servers or web search).
- End the call by itself, either because the request is solved or because a human agent is needed, and tell Architect which one happened so the flow can route the call.
- Return a call summary and usage metrics to Architect when the call ends.

## Contents

1. [How it works](#how-it-works)
2. [Where to host it](#where-to-host-it)
3. [Quick start (local)](#quick-start-local)
4. [Deploy to DigitalOcean App Platform](#deploy-to-digitalocean-app-platform)
5. [Connect Genesys Cloud](#connect-genesys-cloud)
6. [Environment variables](#environment-variables)
7. [Session variables (Architect)](#session-variables-architect)
8. [Ending calls and escalating](#ending-calls-and-escalating)
9. [Genesys Data Actions as tools](#genesys-data-actions-as-tools)
10. [MCP servers and web search](#mcp-servers-and-web-search)
11. [Choosing a vendor](#choosing-a-vendor)
12. [Vendor details](#vendor-details)
13. [Testing](#testing)
14. [Troubleshooting](#troubleshooting)
15. [Project structure](#project-structure)

## How it works

```
Caller ──► Genesys Cloud ──(Audio Connector / WebSocket)──► this connector ──(WebSocket)──► OpenAI / Gemini / Grok
       ◄──                ◄──                               ◄──                            ◄──
```

1. A call reaches the **Call Audio Connector** action in your Architect flow. Genesys opens a WebSocket to the connector and sends the flow's session variables (prompt, vendor, model, voice, tools...).
2. The connector checks the `x-api-key` header, opens a session with the chosen AI vendor and starts streaming audio in both directions (G.711 µ-law, 8 kHz).
3. The model listens, answers, and calls tools when it needs data.
4. When the conversation is over, the model calls one of two built-in functions: `end_conversation_successfully` or `end_conversation_with_escalation`. The connector lets the farewell finish playing, then disconnects.
5. Architect receives output variables (`ESCALATION_REQUIRED`, `CONVERSATION_SUMMARY`, ...) and continues the flow, for example by transferring to a queue.

## Where to host it

**Use DigitalOcean App Platform** (the deployment this project is built and documented for), or any host that runs a long-lived Python process with a public `wss://` endpoint, such as a container service or a VM behind a TLS proxy.

**Vercel is not a good fit.** Genesys keeps one WebSocket open to the connector for the whole call, and the connector keeps another WebSocket open to the AI vendor. Vercel Functions are request/response functions with execution time limits and cannot act as a long-lived WebSocket server, so the connector cannot run there.

## Quick start (local)

**You need:**

- Python 3.10 or newer (the repo pins 3.12 in `.python-version`).
- An API key for at least one vendor: [OpenAI](https://platform.openai.com/api-keys), [Google AI Studio](https://aistudio.google.com/apikey) (Gemini) or [xAI](https://console.x.ai/).
- A Genesys Cloud org with the Audio Connector integration available.

**Steps:**

1. Install the dependencies:

   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create your configuration file from the template and fill in your own values:

   ```bash
   cp .env.local.example .env      # Windows PowerShell: Copy-Item .env.local.example .env
   ```

   At minimum set `GENESYS_API_KEY` (any long random string; you will paste the same value into Genesys), `AI_VENDOR`, and the API key of that vendor. `.env` is gitignored: never commit it.

3. Start the server:

   ```bash
   python oai_middleware.py
   ```

   It listens on `ws://0.0.0.0:8080/audiohook` (change the port with `PORT`). Opening `http://localhost:8080/` in a browser returns `OK`.

4. Optional: run the offline tests (no API keys or network needed): `python tests/smoke_test_vendors.py`.

Genesys Cloud can only reach a public `wss://` URL. For a local trial, expose port 8080 with a tunnel such as ngrok or Cloudflare Tunnel; for real use, deploy it as described below.

## Deploy to DigitalOcean App Platform

### 1. Create a DigitalOcean account

If you don't have one yet, you can get a DigitalOcean account with $200 in free credits here (this also supports my R&D work):

<a href="https://www.digitalocean.com/?refcode=e78e0ec0ec1d&utm_campaign=Referral_Invite&utm_medium=Referral_Program&utm_source=badge"><img src="https://web-platforms.sfo2.cdn.digitaloceanspaces.com/WWW/Badge%201.svg" alt="DigitalOcean Referral Badge" /></a>

### 2. Create the app

1. Fork or push this repository to your own GitHub account.
2. In DigitalOcean, open **Apps** → **Create App** and select your repository.
3. Choose an instance size. The smallest ($5/month) is enough for testing.

### 3. Add the environment variables

Add these in the app settings. Mark every key and secret as **Encrypted**.

| Variable | Value |
|----------|-------|
| `GENESYS_API_KEY` | A long random string. You will enter the same value in the Genesys integration |
| `AI_VENDOR` | `openai`, `gemini` or `grok` (the default vendor for every call) |
| `OPENAI_API_KEY` | Your OpenAI key, if you use OpenAI |
| `GEMINI_API_KEY` | Your Google AI Studio key, if you use Gemini |
| `XAI_API_KEY` | Your xAI key, if you use Grok |

Only the keys of the vendors you use are needed. If you use Genesys Data Actions as tools, also add `GENESYS_CLIENT_ID`, `GENESYS_CLIENT_SECRET` and `GENESYS_REGION` (see [Genesys Data Actions as tools](#genesys-data-actions-as-tools)). Everything else is optional; see [Environment variables](#environment-variables).

Make sure the instance size, environment variables, region and name are set, then click **Create App**:

<img width="976" height="1159" alt="Create App summary" src="https://github.com/user-attachments/assets/7a87e31d-6766-4279-aee0-956e26dc23e4" />

### 4. Set the run command and health check

In the component **Settings**, set the **Run Command** to `python oai_middleware.py`. Without it the deployment fails:

<img width="898" height="1185" alt="Run command setting" src="https://github.com/user-attachments/assets/778872e3-315c-4995-af44-5f7284d95ad8" />

Use these health check settings (the server answers `OK` on `/`):

<img width="561" height="680" alt="Health check settings" src="https://github.com/user-attachments/assets/433d9dc4-da03-4c2f-b2fe-13a367326ca0" />

### 5. Note your app URL

After the deployment finishes, copy the app URL (for example `https://your-app-xxxxx.ondigitalocean.app`). DigitalOcean provides the TLS certificate.

<img width="1438" height="320" alt="App URL" src="https://github.com/user-attachments/assets/df49e434-3d1f-4870-8cf8-e9770e886415" />

## Connect Genesys Cloud

### 1. Create the Audio Connector integration

1. In Genesys Cloud, go to **Admin → Integrations**, add an **Audio Connector** integration.
2. Set the base connection URI to `wss://<your-app-domain>/audiohook` (for DigitalOcean: `wss://your-app-xxxxx.ondigitalocean.app/audiohook`).
3. Set the API key credential to your `GENESYS_API_KEY` value.
4. Save and activate the integration.

### 2. Build the Architect flow

The quickest start is the example inbound call flow in this repo:

- [`architect/DO - Audio Connector - Speech to speech_v21-0.i3InboundFlow`](architect/DO%20-%20Audio%20Connector%20-%20Speech%20to%20speech_v21-0.i3InboundFlow): import it in Architect (**Inbound Call** flows → **Import**).
- [`architect/DO - Audio Connector - Speech to speech.yaml`](architect/DO%20-%20Audio%20Connector%20-%20Speech%20to%20speech.yaml): the same flow as Archy YAML, for Archy or the Platform API.

The flow runs these steps in order:

| Step | What it does |
|------|--------------|
| **Set AI configuration** | Sets the vendor, model, voice, language and maximum answer length in one place (`Flow.aiVendor`, `Flow.aiModel`, `Flow.aiVoice`, `Flow.aiLanguage`, `Flow.aiMaxOutputTokens`). Example: `openai` / `gpt-realtime-2.1` / `coral`, or `grok` / `grok-voice-3.0` / `eve`, or `gemini` / `gemini-3.8-live` / `Kore` |
| **Set general instructions** and **Set Guardrails** | The use-case prompt and the safety rules. They are joined into `AI_SYSTEM_PROMPT` |
| **Call Audio Connector** | Sends the vendor-agnostic session variables (`AI_VENDOR`, `AI_MODEL`, `AI_VOICE`, `LANGUAGE`, `AI_MAX_OUTPUT_TOKENS`, prompts, persona, Data Actions) and receives every output variable |
| **Success path** | Saves the outcome as participant data the agent can see (`AI_Summary`, `AI_CompletionSummary`, `AI_EscalationRequired`, `AI_EscalationReason`, `AI_ConversationDuration`), then transfers to a queue when `ESCALATION_REQUIRED` is `true`, otherwise disconnects |
| **Failure path** | If the connector can't be reached or refuses the call (for example a missing vendor key), saves `AI_ConnectorErrorType` and transfers the caller to the queue instead of dropping the call |

After importing, adapt it to your org:

1. In **Call Audio Connector**, select your Audio Connector integration.
2. In **Set AI configuration**, choose the vendor, model and voice. The model and voice must belong to the selected vendor, and that vendor's API key must be set on the server (see [Models and voices](#models-and-voices) for the options).
3. Rewrite the instructions and guardrails for your use case, and set `AGENT_NAME`, `COMPANY_NAME`, `SUCCESS_PROMPT` and `ESCALATION_PROMPT`.
4. Replace `DATA_ACTION_IDS` and `DATA_ACTION_DESCRIPTIONS` with your own Data Actions (the example IDs only exist in the author's org), or remove them.
5. In the **Transfer to agent** task, select your queue (the example uses `DO_test`).
6. Publish, assign the flow to a phone number, and call it: you should hear the voice agent.

## Environment variables

These are set on the server (`.env` locally, app settings in DigitalOcean). [`.env.local.example`](.env.local.example) lists them all with placeholder values.

### Required

| Variable | Description |
|----------|-------------|
| `GENESYS_API_KEY` | Shared secret that Genesys sends in the `x-api-key` header. Connections with a different key are rejected |
| `AI_VENDOR` | Default vendor: `openai` (default), `gemini` or `grok` (`xai` also works) |
| `OPENAI_API_KEY` / `GEMINI_API_KEY` / `XAI_API_KEY` | API key of each vendor you use |

The server refuses to start if the key of the default `AI_VENDOR` is missing. If a call asks for a vendor whose key is not configured, the call is disconnected with an error that names the missing variable.

### Models and voices

| Variable | Default | Options |
|----------|---------|---------|
| `OPENAI_MODEL` | `gpt-realtime-2.1` | Any `gpt-realtime*` model, or `gpt-live-1` to use GPT-Live |
| `OPENAI_VOICE` | `marin` | `cedar` and the other OpenAI Realtime voices |
| `GEMINI_MODEL` | `gemini-3.8-live` | `gemini-3.8-live-extended-thinking`, `gemini-3.1-flash-live-preview` |
| `GEMINI_VOICE` | `Kore` | 30 voices such as `Puck`, `Charon`, `Zephyr`, `Aoede` ([full list](https://ai.google.dev/gemini-api/docs/speech-generation#voices)) |
| `GROK_MODEL` | `grok-voice-3.0` | `grok-voice-latest` |
| `GROK_VOICE` | `eve` | `ara`, `leo`, `rex`, `sal`, `atlas`, ... (`GET https://api.x.ai/v1/tts/voices`) or a custom voice ID |
| `AI_MODEL` / `AI_VOICE` | not set | Older names; they override the model / voice of the default vendor only |

### Optional tuning

| Variable | Applies to | What it does |
|----------|-----------|--------------|
| `OPENAI_REASONING_EFFORT` | OpenAI Realtime | `minimal`, `low` (default), `medium`, `high`, `xhigh`. Used by the `gpt-realtime-2*` models only |
| `OPENAI_TURN_DETECTION` | OpenAI Realtime | `semantic_vad` (default) or `server_vad` |
| `OPENAI_VAD_EAGERNESS` | OpenAI Realtime | `auto` (default), `low`, `medium`, `high` |
| `OPENAI_NOISE_REDUCTION` | OpenAI Realtime | `near_field` or `far_field` (off by default) |
| `OPENAI_LIVE_DELEGATION_MODEL` | GPT-Live | Model that runs tools and reasoning behind GPT-Live (default `gpt-6-luna`) |
| `OPENAI_LIVE_DELEGATION_REASONING_EFFORT` | GPT-Live | Reasoning effort of that model |
| `OPENAI_LIVE_SERVICE_TIER` | GPT-Live | `auto`, `default`, `flex` or `priority` |
| `GEMINI_THINKING_LEVEL` | Gemini | `low` / `medium` / `high` for extended-thinking, `minimal`..`high` for 3.1 flash live. Ignored by `gemini-3.8-live` |
| `GEMINI_VAD_SILENCE_DURATION_MS` | Gemini | Silence that ends the caller's turn (default `700`) |
| `GEMINI_VAD_PREFIX_PADDING_MS` | Gemini | Speech needed before a turn starts (default `100`) |
| `GEMINI_VAD_START_SENSITIVITY` / `GEMINI_VAD_END_SENSITIVITY` | Gemini | `high` or `low` |
| `GEMINI_CONTEXT_COMPRESSION` | Gemini | Keeps calls longer than 15 minutes working (default `true`) |
| `GEMINI_SESSION_RESUMPTION` | Gemini | Reconnects automatically when Google resets the connection (default `true`) |
| `GEMINI_GOOGLE_SEARCH` | Gemini | Lets the model use Google Search on every call (default `false`) |
| `GEMINI_SUMMARY_MODEL` | Gemini | Text model that writes the call summary (default `gemini-3.5-flash-lite`) |
| `GROK_REASONING_EFFORT` | Grok | `high` (default) or `none` for the fastest answers |
| `GROK_IDLE_TIMEOUT_MS` | Grok | Prompt a silent caller again after this many milliseconds |
| `GROK_SPEECH_SPEED` | Grok | Speaking speed from `0.7` to `1.5` |
| `GROK_KEYTERMS` | Grok | Comma-separated names and product terms to improve speech recognition (max 100) |
| `AGENT_NAME` / `COMPANY_NAME` | All | Default persona names used in the prompt (`AI Assistant` / `Our Company`) |
| `PORT` | All | Listening port (default `8080`) |
| `DEBUG` | All | `true` for debug logs. Debug logs contain prompts and summaries, so keep it `false` in production |

### Genesys Data Actions (optional)

Needed only when the voice agent calls Genesys Cloud Data Actions:

| Variable | Description |
|----------|-------------|
| `GENESYS_CLIENT_ID` / `GENESYS_CLIENT_SECRET` | OAuth client credentials (client credentials grant) allowed to run the data actions |
| `GENESYS_REGION` or `GENESYS_BASE_URL` | Your region (for example `mypurecloud.com`, `usw2.pure.cloud`) or the full API base URL |
| `GENESYS_LOGIN_URL` | Override for the OAuth login URL (default `https://login.<region>`) |
| `GENESYS_ALLOWED_DATA_ACTION_IDS` | Comma-separated allowlist. Only these actions can be exposed, whatever the flow requests |
| `GENESYS_MAX_TOOLS_PER_SESSION` | Maximum data actions loaded per call (default `10`) |
| `GENESYS_MAX_ACTION_CALLS_PER_SESSION` | Maximum data action executions per call (default `10`) |
| `GENESYS_TOOL_OUTPUT_REDACTION_FIELDS` | Comma-separated JSON paths (for example `customer.ssn,account.cardNumber`) removed from results before the model sees them |

Timeouts and retry settings (`GENESYS_HTTP_TIMEOUT_SECONDS`, ...) are in `config.py`.

## Session variables (Architect)

Session variables are set in the **Call Audio Connector** action and override the server defaults for that call.

### Input variables (Architect → connector)

**Most used**

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_SYSTEM_PROMPT` | Instructions for the voice agent | "You are a helpful assistant." |
| `AI_VENDOR` | Vendor for this call: `openai`, `gemini` or `grok` (its key must be on the server) | Server `AI_VENDOR` |
| `LANGUAGE` | Language the agent must speak. A tag such as `es-ES` also improves Grok's speech recognition | Not set |
| `AGENT_NAME` / `COMPANY_NAME` | Names used in the prompt | Server defaults |
| `CUSTOMER_DATA` | Personalization data as `key:value;key:value` | Not set |
| `SUCCESS_PROMPT` | Exact words to say when the call ends successfully | A generated goodbye |
| `ESCALATION_PROMPT` | Exact words to say before a transfer to a human | A generated transfer message |

**Model and voice**

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_MODEL` | Model for this call, for any vendor (for example `gpt-realtime-2.1`, `gpt-live-1`, `gemini-3.8-live`, `grok-voice-3.0`). A model of another vendor is ignored | Server default |
| `AI_VOICE` | Voice for this call, for any vendor (for example `coral`, `Kore`, `eve`). An unknown Gemini voice falls back to the default | Server default |
| `OPENAI_MODEL` / `GEMINI_MODEL` / `GROK_MODEL`, `OPENAI_VOICE` / `GEMINI_VOICE` / `GROK_VOICE` | Vendor-specific alternatives. When set, they take precedence over `AI_MODEL` / `AI_VOICE` for that vendor | Server default |
| `AI_TEMPERATURE` | Randomness of answers. Gemini only; the other vendors have no temperature setting | Model default |
| `AI_MAX_OUTPUT_TOKENS` | Maximum tokens per answer (not supported by Grok) | `inf` |
| `GROK_KEYTERMS` / `GROK_REASONING_EFFORT` | Grok settings for this call | Server values |

**Tools**

| Variable | Description |
|----------|-------------|
| `DATA_ACTION_IDS` | Genesys Data Action IDs to expose as tools, separated by `\|` |
| `DATA_ACTION_DESCRIPTIONS` | One description per ID, same order, separated by `\|` |
| `MCP_TOOLS_JSON` | JSON array of MCP servers / built-in tools (see [MCP servers and web search](#mcp-servers-and-web-search)) |

Older flows keep working: `OPENAI_SYSTEM_PROMPT`, `OPENAI_TEMPERATURE` and `OPENAI_MAX_OUTPUT_TOKENS` are used when the `AI_*` variable is empty, and an `OPENAI_MODEL` value is ignored when the call uses Gemini or Grok, so you can switch an existing flow to another vendor by setting `AI_VENDOR` only.

### Output variables (connector → Architect)

| Variable | Description |
|----------|-------------|
| `ESCALATION_REQUIRED` | `true` if the agent asked for a human agent, otherwise `false` |
| `ESCALATION_REASON` | Why the agent escalated |
| `COMPLETION_SUMMARY` | The agent's one-sentence summary when the call ended successfully |
| `CONVERSATION_SUMMARY` | A 2-3 sentence summary of the whole call |
| `CONVERSATION_DURATION` | Session length in seconds |
| `TOTAL_INPUT_TEXT_TOKENS`, `TOTAL_INPUT_CACHED_TEXT_TOKENS`, `TOTAL_INPUT_AUDIO_TOKENS`, `TOTAL_INPUT_CACHED_AUDIO_TOKENS`, `TOTAL_OUTPUT_TEXT_TOKENS`, `TOTAL_OUTPUT_AUDIO_TOKENS` | Token usage for cost tracking |
| `TOTAL_VOICE_SECONDS` | GPT-Live only: billed voice seconds |

How usage is reported differs per vendor: OpenAI Realtime and Gemini give a full text / audio / cached breakdown; Grok only reports totals, which are placed in the two `*_AUDIO_TOKENS` fields; GPT-Live bills voice per second (`TOTAL_VOICE_SECONDS`) and its text token fields count the tool model and the summary.

## Ending calls and escalating

Every session includes two functions the model calls to finish the call:

| Function | When the model calls it | What happens |
|----------|------------------------|--------------|
| `end_conversation_successfully` | The request is solved **and** the caller confirms they need nothing else | The agent says goodbye (`SUCCESS_PROMPT` if set), then the call is disconnected with `ESCALATION_REQUIRED=false` and `COMPLETION_SUMMARY` filled |
| `end_conversation_with_escalation` | The caller asks for a person, is clearly unhappy, or the task can't be done | The agent announces the transfer (`ESCALATION_PROMPT` if set), then the call is disconnected with `ESCALATION_REQUIRED=true` and `ESCALATION_REASON` filled |

In both cases the connector waits until the farewell has finished playing before it disconnects, and the summary is generated while the farewell plays so the caller doesn't wait. Grok speaks `SUCCESS_PROMPT` / `ESCALATION_PROMPT` word for word; the other vendors are instructed to say them exactly.

In Architect, branch on `ESCALATION_REQUIRED` after the Call Audio Connector action: transfer to a queue when it is `true`, end or wrap up the call otherwise.

## Genesys Data Actions as tools

When `DATA_ACTION_IDS` is set, the connector reads each data action's input schema from Genesys Cloud and offers it to the model as a function (for example `genesys_data_action_get_ticket`). When the model calls it, the connector runs the data action and returns the result to the model, which explains it to the caller. If the model calls several tools at once, they run in parallel.

**Setup:**

1. Create an OAuth client (client credentials) in Genesys Cloud with permission to execute the data actions, and set `GENESYS_CLIENT_ID`, `GENESYS_CLIENT_SECRET` and `GENESYS_REGION` on the server.
2. Test each data action in Genesys Cloud first.
3. In the Call Audio Connector action, set:
   - `DATA_ACTION_IDS`: the action IDs separated by `|`. IDs look like `custom_-_9e9b11f4-ddfc-40d1-a7d3-d24f3815e818`.
   - `DATA_ACTION_DESCRIPTIONS`: one description per action, **in the same order**, separated by `|`.

The descriptions are what the model reads to decide when to use each tool, so explain what the action does and when to use it. To force a fixed value for an input parameter, say so in the description (the first description below does this).

**Example:**

```
DATA_ACTION_IDS = custom_-_9e9b11f4-ddfc-40d1-a7d3-d24f3815e818|custom_-_37be4e2a-805b-4b68-a7df-0fd2768c27b8|custom_-_e4744e9d-e3f2-4ad6-acf1-678717230e25|custom_-_ee93134c-f43e-4278-a5c3-5f98f9dcf4e1

DATA_ACTION_DESCRIPTIONS = Searches knowledge base articles to answer general FAQ questions. Use this to provide RAG responses. For the KBId input param, use dc8de859-0102-4ebd-b216-f3f31e1c78c6, for minConfidence use 0.6 and for maxArticles use 3|Retrieves complete ticket details including customer information, journey stations, departure schedule, and fare class using the ticket reference number|Checks availability for ticket modifications and returns alternative departure times and fare class options with associated change fees|Updates the ticket booking with the new departure date/time and fare class after the customer confirms
```

With this configuration, a caller asking "What's the baggage policy?" triggers the knowledge search, and a caller giving a ticket number triggers the ticket lookup.

**Safety controls:**

- Only actions listed in `DATA_ACTION_IDS` are exposed, and `GENESYS_ALLOWED_DATA_ACTION_IDS` on the server can restrict them further.
- `GENESYS_MAX_ACTION_CALLS_PER_SESSION` and `GENESYS_MAX_TOOLS_PER_SESSION` limit usage per call.
- `GENESYS_TOOL_OUTPUT_REDACTION_FIELDS` removes sensitive fields before the model sees them. If redaction fails, the result is withheld instead of being sent unredacted.
- Logs contain action IDs and argument names, never argument values or results.

**If the agent picks the wrong tool**, check that IDs and descriptions are in the same order, and improve the descriptions (include the phrases callers use).

## MCP servers and web search

Set the `MCP_TOOLS_JSON` session variable to a JSON array to give the agent remote [MCP](https://platform.openai.com/docs/guides/tools-remote-mcp) servers or vendor built-in tools. Leave it empty to disable. Example:

```json
[
  {"type": "mcp", "server_label": "deepwiki", "server_url": "https://mcp.deepwiki.com/mcp", "require_approval": "never"},
  {"type": "web_search"}
]
```

What each vendor uses:

| Vendor | Supported entries |
|--------|-------------------|
| OpenAI Realtime | Passed as-is: `mcp` and any built-in tool type the Realtime API accepts |
| GPT-Live | Passed to its tool model. Approval prompts can't be answered on a phone call and are rejected, so use `"require_approval": "never"` |
| Grok | `mcp`, `web_search`, `x_search`, `file_search` (`web_search_preview` is treated as `web_search`); other types are skipped |
| Gemini | `web_search` / `google_search` turn on Google Search; MCP entries are skipped (not supported by Gemini Live) |

If an MCP entry contains an `authorization` token, treat the variable as a secret. The connector never logs its contents.

## Choosing a vendor

| | OpenAI Realtime | OpenAI GPT-Live | Gemini Live | xAI Grok Voice |
|---|---|---|---|---|
| **Telephony audio** | Native 8 kHz µ-law | Native 8 kHz µ-law | Converted from 24 kHz | Native 8 kHz µ-law |
| **Turn taking** | Voice activity detection (semantic or silence based) | Full duplex: the model decides when to speak | Voice activity detection | Voice activity detection |
| **Tools** | Functions, MCP, built-ins | Run by a separate tool model | Functions, Google Search | Functions, web / X / file search, MCP |
| **Reasoning control** | `OPENAI_REASONING_EFFORT` | Tool model reasoning | `GEMINI_THINKING_LEVEL` (some models) | `GROK_REASONING_EFFORT` |
| **Exact farewell text** | Instructed | Instructed | Instructed | Spoken word for word |
| **Interruption handling** | Yes | Partial (see [Vendor details](#vendor-details)) | Yes | Yes |
| **Usage metrics** | Detailed tokens | Voice seconds + tokens | Detailed tokens | Total tokens |

Because each call can choose its vendor with the `AI_VENDOR` session variable, you can compare vendors from Architect without redeploying.

## Vendor details

The xAI Voice Agent API uses the same protocol as OpenAI Realtime, so both share one WebSocket client (`openai_client.py`, with Grok differences in `grok_client.py`), and GPT-Live (`gpt_live_client.py`) reuses its transport. Gemini uses Google's official `google-genai` SDK.

**OpenAI Realtime**

- Connects to `wss://api.openai.com/v1/realtime?model=<model>` (GA interface).
- Audio stays in µ-law, so no conversion is needed.
- On an interruption the connector stops playback in Genesys and tells OpenAI how much of the answer the caller actually heard, so the conversation history stays accurate.
- The call summary is generated as a separate text response that doesn't affect the conversation.

**OpenAI GPT-Live** (used when the OpenAI model is `gpt-live-*`)

- Connects to `wss://api.openai.com/v1/live/sessions`.
- GPT-Live talks and listens at the same time and decides itself when to speak. Tools and reasoning run on a separate model (`OPENAI_LIVE_DELEGATION_MODEL`).
- The summary is generated from the call transcript with one Responses API request (not stored by OpenAI).
- Limitation: GPT-Live sends no interruption event, so up to about one second of already-queued audio can still play after the caller interrupts.

**xAI Grok**

- Connects to `wss://api.x.ai/v1/realtime?model=<model>`.
- Audio stays in µ-law.
- `LANGUAGE` (as a tag like `en-US`) and `GROK_KEYTERMS` improve speech recognition.
- With `SUCCESS_PROMPT` / `ESCALATION_PROMPT` set, the farewell text is spoken exactly and can't be interrupted.

**Google Gemini Live**

- Uses the `google-genai` SDK. The API key travels in a request header, never in a URL.
- Caller audio is sent at 8 kHz; the model's 24 kHz audio is converted to 8 kHz µ-law with a high-quality streaming resampler.
- Google resets Live connections about every 10 minutes; the connector resumes the session automatically, and context compression keeps calls longer than 15 minutes working.
- `gemini-3.8-live` runs tools in the background while it keeps talking; the call-ending functions wait for completion. Models that don't accept a setting (for example thinking level on `gemini-3.8-live`) never receive it.
- Live models only speak, so the call summary is written by `GEMINI_SUMMARY_MODEL` from the transcript.

## Testing

`tests/smoke_test_vendors.py` checks the whole connector offline: it starts local mock versions of the OpenAI, GPT-Live and Grok servers, fakes the Gemini SDK session and a Genesys connection, and uses fake API keys, so it never contacts a real service.

```bash
python tests/smoke_test_vendors.py
```

It covers session setup, audio in both directions, parallel tool calls, interruptions, farewells and disconnects, summaries, usage metrics, Gemini reconnection, per-call vendor selection, and the server handshake (API key check, health check, open / ping / close). Expected output ends with `7/7 passed`.

After deploying, make one real test call per vendor you plan to use.

## Troubleshooting

| Symptom | What to check |
|---------|---------------|
| Server stops at startup with `... not found in environment variables` | The key of the default `AI_VENDOR` (or `GENESYS_API_KEY`) is not set |
| Genesys can't connect, logs show `Invalid API Key` or `Missing 'x-api-key'` | The API key in the Genesys integration must equal `GENESYS_API_KEY` |
| Logs show `Path mismatch` | The integration URI must end with `/audiohook` |
| Call drops right away with `AI_VENDOR=... requested but ..._API_KEY is not configured` | The flow selected a vendor whose key is not on the server |
| Call drops with `Failed to connect to <vendor>` | Wrong or unauthorized vendor key, or the account has no access to that model |
| The agent never ends the call | Make the prompt clear about when the conversation is finished; the model decides when to call the end functions |
| The agent picks the wrong Data Action | Check the order of `DATA_ACTION_IDS` / `DATA_ACTION_DESCRIPTIONS` and improve the descriptions |
| `Data actions requested but GENESYS_CLIENT_ID / GENESYS_CLIENT_SECRET are not configured` | Set the OAuth client variables on the server |

Set `DEBUG=true` temporarily for more detail, and turn it off again afterwards.

## Project structure

| File | Purpose |
|------|---------|
| `oai_middleware.py` | Entry point: WebSocket server, `x-api-key` check, health check |
| `audio_hook_server.py` | Genesys AudioHook protocol for one call: vendor selection, audio playback, disconnect and output variables |
| `openai_client.py` | OpenAI Realtime client (also the base for Grok and GPT-Live) |
| `gpt_live_client.py` | OpenAI GPT-Live client |
| `grok_client.py` | xAI Grok Voice client |
| `gemini_client.py` | Google Gemini Live client |
| `voice_client_base.py` | Shared behavior: tools, farewell, playback waits, usage metrics |
| `call_control.py` | The two call-ending functions |
| `genesys_actions.py` | Genesys Data Actions as tools |
| `mcp_tools.py` | Parsing of `MCP_TOOLS_JSON` |
| `config.py` | Environment variables and defaults |
| `utils.py` | Audio helpers, prompt building, WebSocket helpers |
| `rate_limiter.py` | Keeps traffic to Genesys within AudioHook rate limits |
| `tests/smoke_test_vendors.py` | Offline smoke tests |
| `architect/` | Example Architect inbound call flow (Architect export and Archy YAML) |
| `.env.local.example` | Template for your `.env` (placeholders only) |
