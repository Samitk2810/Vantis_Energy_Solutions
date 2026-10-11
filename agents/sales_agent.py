import json
from pathlib import Path
from typing import Literal, Optional
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from db.repo import can_persist, create_callback, save_lead
from agents.state import AgentState, merge_fields
from tools.service_area import check_service_area
from tools.validators import clean_email, clean_phone

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

EXTRACTION_PROMPT = """You extract sales-lead details for Vantis Energy Solutions (rooftop solar) from a customer chat.

Rules:
- Extract ONLY what the customer actually stated. Never guess or infer. Use null for anything not stated.
- Look at the customer's latest message first. Use the recent chat only to understand short answers (for example "yes" or "Kota" in reply to a question).
- Do not repeat values that are already known unless the customer changed them.
- phone: copy exactly what the customer wrote. Do not fix or complete it.
- city: the city of the property. locality: area, sector or neighbourhood within the city.
- property_type: "residential" (home, house, villa, flat) or "commercial" (shop, office, clinic, warehouse, business). ownership: "owned" or "rented".
- intent: "quote" if they want a quote, installation, site visit or callback; "sizing" if they ask what size or cost suits them; otherwise "general".
- system_size_kw: only if they name a size in kW. package: only if they name a Vantis package.
- monthly_bill: in rupees as a number. monthly_units: electricity units per month as a number.
- timeline_months: when they want installation, converted to months: "asap", "now", "this month" = 1; "next month" = 1; "in 3 months" = 3; "next year" = 12. null if not stated.
- wants_callback: true only if they ask to be called, contacted or to speak to the sales team. callback_time: their preferred time exactly as stated, else null.
- budget: as stated, else null."""

REPLY_PROMPT = """You write the short lead-collection part of a reply from the Vantis Energy Solutions assistant. You are given JSON facts. Write 1 to 2 short, friendly sentences in plain text.

Rules:
- Only say something was done if the facts say so.
- If saved_new is true, say you have noted their details and passed them to the sales team.
- If callback_saved is true, say you have noted their callback request and the sales team will follow up. Never promise a specific time, date or call duration.
- If location_status is "unverified" and saved_new is true, also say the sales team will verify service availability for their area.
- If invalid_phone is true, say the number doesn't look right and ask for a 10-digit mobile number.
- If ask is not null, finish by asking exactly that question once, in natural words. Ask nothing else. If ask is null, do not ask any question.
- Never mention scores, lead status, internal systems, or the word "database".
- Never state prices, savings, subsidies, warranty periods or installation dates."""

INTENT_RANK = {"general": 1, "sizing": 2, "quote": 3}

QUESTIONS = {
    "city": "Which city is the property in?",
    "property_type": "Is this for a home or a business?",
    "business_type": "What kind of business is it?",
    "ownership": "Do you own the property or is it rented?",
    "phone": "What's the best phone number to reach you on?",
    "energy": "Roughly what is your monthly electricity bill?",
    "timeline": "When are you hoping to get the installation done?",
    "name": "May I have your name?",
}


class LeadExtraction(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    city: Optional[str] = None
    locality: Optional[str] = None
    property_type: Optional[Literal["residential", "commercial"]] = None
    business_type: Optional[str] = None
    ownership: Optional[Literal["owned", "rented"]] = None
    intent: Optional[Literal["quote", "sizing", "general"]] = None
    system_size_kw: Optional[float] = None
    package: Optional[str] = None
    monthly_bill: Optional[float] = None
    monthly_units: Optional[float] = None
    timeline_months: Optional[float] = None
    budget: Optional[str] = None
    wants_callback: bool = False
    callback_time: Optional[str] = None


extractor = llm.with_structured_output(LeadExtraction)


def extract_fields(state: AgentState) -> LeadExtraction:
    msgs = state["messages"]
    recent = "\n".join(
        f"{'Customer' if m.type == 'human' else 'Assistant'}: {m.content}" for m in msgs[-7:-1]
    )
    known = {k: v for k, v in (state.get("lead_fields") or {}).items() if v}
    prompt = (
        f"Already known: {known or 'nothing'}\n"
        f"Recent chat:\n{recent or '(none)'}\n"
        f"Customer's latest message: {msgs[-1].content}"
    )
    try:
        return extractor.invoke([SystemMessage(content=EXTRACTION_PROMPT), HumanMessage(content=prompt)])
    except Exception:
        return LeadExtraction()


def missing_fields(f: dict) -> list:
    """Fields still worth asking for, in priority order."""
    need = []
    if not f.get("city"):
        need.append("city")
    if not f.get("property_type"):
        need.append("property_type")
    if f.get("property_type") == "commercial":
        if not f.get("business_type"):
            need.append("business_type")
        if not f.get("ownership"):
            need.append("ownership")
    if not (f.get("phone") or f.get("email")):
        need.append("phone")
    if not (f.get("monthly_bill") or f.get("monthly_units")):
        need.append("energy")
    if f.get("timeline_months") is None:
        need.append("timeline")
    if not f.get("name"):
        need.append("name")
    return need


def _fallback_reply(facts: dict) -> str:
    parts = []
    if facts["saved_new"]:
        parts.append("Thanks, I've noted your details and passed them to our sales team.")
        if facts["location_status"] == "unverified":
            parts.append("They'll also verify service availability for your area.")
    if facts["callback_saved"]:
        parts.append("I've noted your callback request and the sales team will follow up.")
    if facts["invalid_phone"]:
        parts.append("That phone number doesn't look right. Could you share a 10-digit mobile number?")
    elif facts["ask"]:
        parts.append(facts["ask"])
    return " ".join(parts)


def compose_reply(facts: dict) -> str:
    try:
        text = llm.invoke(
            [SystemMessage(content=REPLY_PROMPT), HumanMessage(content=json.dumps(facts))]
        ).content.strip()
        return text or _fallback_reply(facts)
    except Exception:
        return _fallback_reply(facts)


def lead_node(state: AgentState):
    existing_reply = state.get("reply") or ""
    old = state.get("lead_fields") or {}
    ex = extract_fields(state)

    new = {
        k: v
        for k, v in ex.model_dump(exclude={"wants_callback", "callback_time"}).items()
        if v not in (None, "")
    }

    # validate contact details
    invalid_phone = False
    if "phone" in new:
        phone = clean_phone(new["phone"])
        if phone:
            new["phone"] = phone
        else:
            invalid_phone = True
            del new["phone"]
    if "email" in new:
        email = clean_email(new["email"])
        if email:
            new["email"] = email
        else:
            del new["email"]
    if "city" in new:
        new["city"] = check_service_area(new["city"])[1]

    # intent: a callback request is a quote-level signal; never downgrade intent
    if ex.wants_callback:
        new["intent"] = "quote"
    if "intent" in new and INTENT_RANK.get(old.get("intent"), 0) >= INTENT_RANK[new["intent"]]:
        del new["intent"]

    merged = merge_fields(old, new)
    location_status = state.get("location_status")
    if merged.get("city"):
        location_status = check_service_area(merged["city"])[0]

    update = {"lead_fields": new}
    if location_status:
        update["location_status"] = location_status

    # save only when there is a contact method AND a requirement
    saved_new = callback_saved = False
    if can_persist(merged):
        lead = save_lead(state.get("session_id"), merged, location_status)
        saved_new = state.get("lead_id") is None
        update["lead_id"] = lead["id"]
        if ex.wants_callback:
            create_callback(lead["id"], ex.callback_time, "Requested in chat")
            callback_saved = True

    # decide the single next question
    asked = list(state.get("asked_fields") or [])
    if invalid_phone and "phone" in asked:
        asked.remove("phone")
    todo = missing_fields(merged)
    ask = None
    if invalid_phone or (ex.wants_callback and not callback_saved):
        ask = "phone"
    elif not existing_reply.rstrip().endswith("?"):      # knowledge reply didn't already ask
        ask = next((f for f in todo if f not in asked), None)
    if ask:
        asked.append(ask)
    update["asked_fields"] = asked
    update["awaiting"] = "lead_details" if ask else None

    facts = {
        "saved_new": saved_new,
        "callback_saved": callback_saved,
        "invalid_phone": invalid_phone,
        "location_status": location_status,
        "ask": QUESTIONS[ask] if ask else None,
    }
    if saved_new or callback_saved or invalid_phone or ask:
        text = compose_reply(facts)
    else:
        text = "" if existing_reply else "Thanks! Is there anything else I can help you with?"
    update["reply"] = (existing_reply + "\n\n" + text).strip() if text else existing_reply
    return update