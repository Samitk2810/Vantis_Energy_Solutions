from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Intent = Literal["knowledge", "lead", "support", "smalltalk", "offtopic"]
Awaiting = Literal["lead_details", "support_details"]
LocationStatus = Literal["supported", "unverified"]


def merge_fields(old: Optional[dict], new: Optional[dict]) -> dict:
    """Merge partial field updates. Empty values never overwrite existing ones."""
    merged = dict(old or {})
    for key, value in (new or {}).items():
        if value not in (None, "", []):
            merged[key] = value
    return merged


class AgentState(TypedDict, total=False):
    session_id: str
    messages: Annotated[list[AnyMessage], add_messages]  # full chat history
    intents: list[Intent]                                # set by orchestrator each turn
    lead_fields: Annotated[dict, merge_fields]           # name, phone, email, city, ...
    support_fields: Annotated[dict, merge_fields]        # name, contact, project, issue
    lead_id: Optional[int]                               # set once the lead is saved
    awaiting: Optional[Awaiting]                         # what we last asked the user for
    location_status: Optional[LocationStatus]            # from the city check
    offtopic_count: int                                  # consecutive off-topic messages
    reply: str                                           # final reply for this turn
    asked_fields: list[str]                              # lead questions already asked