import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import logger


@dataclass
class MCPToolContext:
    tools: List[Dict[str, Any]]
    instructions: Optional[str]


def _summarize_tool(entry: Dict[str, Any]) -> str:
    tool_type = entry.get("type", "mcp")
    if tool_type == "mcp":
        label = entry.get("server_label") or entry.get("server_name") or entry.get("name") or "mcp_server"
        server_block = entry.get("server") if isinstance(entry.get("server"), dict) else {}
        url = entry.get("server_url") or entry.get("url") or server_block.get("url")
        return f"- {label} (remote MCP server at {url or 'custom transport'})"
    name = entry.get("name") or tool_type
    return f"- {name} (built-in tool: {tool_type})"


def load_mcp_tool_context(raw_json: Optional[str], session_logger=None) -> Optional[MCPToolContext]:
    """
    Build an MCP tool context from the Genesys MCP_TOOLS_JSON session variable.
    """
    log = session_logger or logger
    blob = (raw_json or "").strip()
    if not blob:
        return None

    source = "MCP_TOOLS_JSON session variable"
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as exc:
        # No content preview: MCP entries can carry server authorization tokens.
        log.error("[MCP] Failed to parse MCP tool JSON from %s at line %s column %s", source, exc.lineno, exc.colno)
        return None

    if not isinstance(parsed, list):
        log.error("[MCP] MCP tool configuration must be a JSON array, got %s from %s", type(parsed).__name__, source)
        return None

    tools: List[Dict[str, Any]] = []
    summaries: List[str] = []

    for idx, entry in enumerate(parsed):
        if not isinstance(entry, dict):
            log.warning("[MCP] Ignoring entry %s from %s because it is not an object", idx, source)
            continue

        tool_type = entry.get("type")
        if not tool_type:
            log.warning("[MCP] Ignoring entry %s from %s because it does not define a type", idx, source)
            continue

        if tool_type == "mcp":
            has_connection = any(key in entry for key in ("server_url", "url", "server"))
            if not has_connection:
                log.warning("[MCP] Skipping MCP entry %s from %s because it lacks server_url/server info", idx, source)
                continue

        tools.append(entry)
        summaries.append(_summarize_tool(entry))

    if not tools:
        log.warning("[MCP] No valid MCP tool entries found in %s", source)
        return None

    instruction_lines = [
        "Remote Model Context Protocol (MCP) integrations are enabled for this conversation.",
        "When you need information or actions from these external systems, call the appropriate MCP tool instead of guessing.",
    ]
    if summaries:
        instruction_lines.append("Registered MCP endpoints:")
        instruction_lines.extend(summaries)

    log.info("[MCP] Loaded %s MCP tool definitions from %s", len(tools), source)
    return MCPToolContext(tools=copy.deepcopy(tools), instructions="\n".join(instruction_lines))
