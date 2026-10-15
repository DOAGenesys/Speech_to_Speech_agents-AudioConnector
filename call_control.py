"""
Call-control tools shared by every vendor client.

The model ends the AudioHook session by calling one of these two functions. Each client
converts the definitions to its vendor's tool format and uses ``resolve_call_control`` to
decide what to return to the model and what to say before Genesys disconnects.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

END_SUCCESS = "end_conversation_successfully"
END_ESCALATION = "end_conversation_with_escalation"

_LEGACY_ALIASES = {
    "end_call": END_SUCCESS,
    "handoff_to_human": END_ESCALATION,
}

CALL_CONTROL_GUIDANCE = f"""[CALL CONTROL]
You have two functions that end the phone call:
- `{END_SUCCESS}`: call it only when the caller's request is fully resolved AND the caller confirms they need nothing else. Put a one-sentence summary of what was accomplished in `summary`.
- `{END_ESCALATION}`: call it when the caller asks for a human, is clearly dissatisfied, or the task cannot be completed. Put the reason in `reason`.
Do not end the call prematurely. Do not say goodbye before calling the function; a closing message is delivered after the function runs."""


def call_control_function_tools() -> List[Dict[str, Any]]:
    """Definitions in the flat function-tool format used by OpenAI Realtime, GPT-Live (Responses) and Grok."""
    return [
        {
            "type": "function",
            "name": END_SUCCESS,
            "description": (
                "Gracefully end the phone call when the caller confirms their needs are met. "
                "Provide a short summary of the completed task in the `summary` field."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One-sentence summary of what was accomplished before ending the call."
                    }
                },
                "required": ["summary"],
                "additionalProperties": False
            }
        },
        {
            "type": "function",
            "name": END_ESCALATION,
            "description": (
                "End the phone call and request a warm transfer to a human agent when the caller asks for a person, "
                "is dissatisfied, or the task cannot be completed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why escalation is needed (e.g., customer requested human, policy restriction, unable to authenticate)."
                    }
                },
                "required": ["reason"],
                "additionalProperties": False
            }
        }
    ]


def canonical_call_control_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    name = _LEGACY_ALIASES.get(name, name)
    return name if name in (END_SUCCESS, END_ESCALATION) else None


def is_call_control(name: Optional[str]) -> bool:
    return canonical_call_control_name(name) is not None


@dataclass
class CallControlResolution:
    action: str
    disconnect_reason: str
    info: str
    output_payload: Dict[str, Any]
    exact_farewell: Optional[str] = None
    farewell_instruction: str = ""

    @property
    def is_escalation(self) -> bool:
        return self.action == END_ESCALATION


def resolve_call_control(
    name: str,
    args: Optional[Dict[str, Any]],
    success_prompt: Optional[str] = None,
    escalation_prompt: Optional[str] = None,
) -> Optional[CallControlResolution]:
    """Map a call-control function call to its tool output and the farewell to deliver."""
    action = canonical_call_control_name(name)
    if action is None:
        return None
    args = args or {}

    if action == END_SUCCESS:
        summary = args.get("summary") or args.get("note") or "Customer confirmed the request was completed."
        exact = (success_prompt or "").strip() or None
        return CallControlResolution(
            action=action,
            disconnect_reason="completed",
            info=summary,
            output_payload={"result": "ok", "action": action, "summary": summary},
            exact_farewell=exact,
            farewell_instruction=(
                f'Say exactly this to the caller, then stop speaking: "{exact}"' if exact
                else "Confirm the task is wrapped up and thank the caller in one short sentence, in the language of the conversation."
            ),
        )

    reason = args.get("reason") or "Caller requested escalation"
    exact = (escalation_prompt or "").strip() or None
    return CallControlResolution(
        action=action,
        disconnect_reason="transfer",
        info=reason,
        output_payload={"result": "ok", "action": action, "reason": reason},
        exact_farewell=exact,
        farewell_instruction=(
            f'Say exactly this to the caller, then stop speaking: "{exact}"' if exact
            else ("Let the caller know a live agent will take over and reassure them help is coming, in one short sentence, "
                  "in the language of the conversation.")
        ),
    )
