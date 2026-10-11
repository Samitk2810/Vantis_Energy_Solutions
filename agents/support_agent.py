import json
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel

from db.repo import create_support_request
from agents.state import AgentState, merge_fields
from tools.validators import clean_email, clean_phone

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)


EXTRACTION_PROMPT = """You extract support-request details for Vantis Energy Solutions (rooftop solar) from a chat with an existing customer.

Rules:
- Extract ONLY what the customer actually stated. Never guess. Use null for anything not stated.
- Look at the customer's latest message first. Use the recent chat only to understand short answers.
- Do not repeat values that are already known unless the customer changed them.
- name: the customer's name.
- contact: a phone number or email exactly as the customer wrote it. Do not fix or complete it.
- project: details that identify their installation, such as system size, location, installation date or project reference, as stated.
- issue: what is wrong or what they need (for example inverter beeping, no power output, maintenance request), as stated. If they describe a new, different problem, return the new one."""

REPLY_PROMPT = """You write a short reply from the Vantis Energy Solutions assistant to an existing customer who needs support. You are given JSON facts. Write 1 to 3 short, friendly sentences in plain text.

Rules:
- Be empathetic and calm. Do not diagnose the problem or give technical fixes.
- If ticket_created is true, say the request has been logged and give ticket_id as their reference number. Say the support team will follow up. Never promise a visit, a fix, a specific time or a date.
- If invalid_contact is true, say the contact detail doesn't look right and ask for a 10-digit mobile number or a valid email.
- If missing is not empty, ask for those details together in one natural sentence. Ask for nothing else. If missing is empty, ask no question.
- Never state warranty periods, prices or installation dates.
- Never say a human has done something unless the facts say so."""

LABELS = {
    "name": "your name",
    "contact": "a phone number or email to reach you",
    "project": "your project details (system size, location or installation date)",
    "issue": "a description of the issue",
}


class SupportExtraction(BaseModel):
    name: Optional[str] = None
    contact: Optional[str] = None
    project: Optional[str] = None
    issue: Optional[str] = None


extractor = llm.with_structured_output(SupportExtraction)


def extract_fields(state: AgentState) -> SupportExtraction:
    msgs = state["messages"]
    recent = "\n".join(
        f"{'Customer' if m.type == 'human' else 'Assistant'}: {m.content}" for m in msgs[-7:-1]
    )
    known = {k: v for k, v in (state.get("support_fields") or {}).items()
             if v and k in LABELS}
    prompt = (
        f"Already known: {known or 'nothing'}\n"
        f"Recent chat:\n{recent or '(none)'}\n"
        f"Customer's latest message: {msgs[-1].content}"
    )
    try:
        return extractor.invoke([SystemMessage(content=EXTRACTION_PROMPT), HumanMessage(content=prompt)])
    except Exception:
        return SupportExtraction()


def _fallback_reply(facts: dict) -> str:
    parts = []
    if facts["ticket_created"]:
        parts.append(
            f"I'm sorry about the trouble. I've logged your request (reference #{facts['ticket_id']}) "
            "and our support team will follow up."
        )
    if facts["invalid_contact"]:
        parts.append("That contact detail doesn't look right. Could you share a 10-digit mobile number or a valid email?")
    elif facts["missing"]:
        parts.append("Could you share " + ", ".join(facts["missing"]) + "?")
    return " ".join(parts)


def compose_reply(facts: dict) -> str:
    try:
        text = llm.invoke(
            [SystemMessage(content=REPLY_PROMPT), HumanMessage(content=json.dumps(facts))]
        ).content.strip()
        return text or _fallback_reply(facts)
    except Exception:
        return _fallback_reply(facts)


def support_node(state: AgentState):
    existing_reply = state.get("reply") or ""
    old = state.get("support_fields") or {}
    ex = extract_fields(state)

    new = {k: v for k, v in ex.model_dump().items() if v not in (None, "")}

    # validate contact: phone or email
    invalid_contact = False
    if "contact" in new:
        raw = new["contact"]
        clean = clean_email(raw) if "@" in raw else clean_phone(raw)
        if clean:
            new["contact"] = clean
        else:
            invalid_contact = True
            del new["contact"]

    merged = merge_fields(old, new)
    missing_keys = [k for k in LABELS if not merged.get(k)]

    # log a ticket once everything is collected (and only once per distinct issue)
    ticket_created, ticket_id = False, old.get("ticket_id")
    if not missing_keys and merged.get("issue") != old.get("ticket_issue"):
        req = create_support_request(
            state.get("session_id"), merged["name"], merged["contact"], merged["project"], merged["issue"]
        )
        ticket_id = req["id"]
        ticket_created = True
        new["ticket_id"] = ticket_id
        new["ticket_issue"] = merged["issue"]

    update = {"support_fields": new}
    still_collecting = bool(missing_keys) or invalid_contact
    update["awaiting"] = "support_details" if still_collecting else None

    facts = {
        "ticket_created": ticket_created,
        "ticket_id": ticket_id if ticket_created else None,
        "invalid_contact": invalid_contact,
        "missing": [] if existing_reply.rstrip().endswith("?") and not invalid_contact
                   else [LABELS[k] for k in missing_keys],
    }
    if ticket_created or invalid_contact or facts["missing"]:
        text = compose_reply(facts)
    else:
        text = "" if existing_reply else "Your request is already logged, and our support team will follow up."
    update["reply"] = (existing_reply + "\n\n" + text).strip() if text else existing_reply
    return update