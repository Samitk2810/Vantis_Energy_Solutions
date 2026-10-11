import os
import secrets
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

from agents.graph import graph, run_turn
from db import repo
from db.models import engine
from tools.validators import clean_phone

DASHBOARD_API_KEY = os.getenv("DASHBOARD_API_KEY", "change-me")
FRONTEND = Path(__file__).resolve().parent / "frontend"

app = FastAPI(title="Vantis Sales & Lead Qualification Agent", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------- auth (dashboard endpoints only) ----------

def require_key(x_api_key: str = Header(default="")):
    if not secrets.compare_digest(x_api_key, DASHBOARD_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------- schemas ----------

class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, max_length=64)
    message: str = Field(min_length=1, max_length=1000)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    lead_id: Optional[int] = None


class WhatsAppMessage(BaseModel):
    from_number: str = Field(alias="from", min_length=5, max_length=20)
    text: str = Field(min_length=1, max_length=1000)


class LeadFields(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    property_type: Optional[str] = None
    business_type: Optional[str] = None
    ownership: Optional[str] = None
    intent: Optional[str] = None
    system_size_kw: Optional[float] = None
    package: Optional[str] = None
    monthly_bill: Optional[float] = None
    monthly_units: Optional[float] = None
    timeline_months: Optional[float] = None
    budget: Optional[str] = None
    notes: Optional[str] = None


class CallbackRequest(BaseModel):
    lead_id: int
    preferred_time: Optional[str] = Field(default=None, max_length=120)
    note: Optional[str] = Field(default=None, max_length=500)


def _transcript(session_id: Optional[str]) -> list[dict]:
    if not session_id:
        return []
    try:
        values = graph.get_state({"configurable": {"thread_id": session_id}}).values
    except Exception:
        return []
    return [
        {"role": "customer" if m.type == "human" else "assistant", "text": m.content}
        for m in values.get("messages", [])
        if m.content
    ]


# ---------- customer-facing ----------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or uuid.uuid4().hex
    result = run_turn(session_id, req.message.strip())
    lead = result.get("lead")
    return ChatResponse(session_id=session_id, reply=result["reply"], lead_id=lead["id"] if lead else None)


@app.post("/webhook/whatsapp")
def whatsapp_webhook(msg: WhatsAppMessage):
    """Simulated WhatsApp inbound message. The sender's number is treated as known contact info."""
    phone = clean_phone(msg.from_number)
    session_id = f"wa:{phone or msg.from_number}"
    result = run_turn(session_id, msg.text.strip(), known_fields={"phone": phone} if phone else None)
    return {"to": msg.from_number, "text": result["reply"]}


@app.post("/callback", status_code=201)
def request_callback(req: CallbackRequest):
    if repo.get_lead(req.lead_id) is None:
        raise HTTPException(status_code=404, detail="Lead not found")
    return repo.create_callback(req.lead_id, req.preferred_time, req.note)


# ---------- dashboard (API key required) ----------

@app.get("/leads", dependencies=[Depends(require_key)])
def list_leads(
    status: Optional[str] = Query(default=None, pattern="^(?i:hot|warm|cold)$"),
    city: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    return repo.list_leads(status=status, city=city, limit=limit, offset=offset)


@app.post("/leads", status_code=201, dependencies=[Depends(require_key)])
def create_lead(fields: LeadFields):
    data = fields.model_dump(exclude_none=True)
    if not repo.can_persist(data):
        raise HTTPException(
            status_code=422,
            detail="A lead needs a valid phone or email plus a requirement (property type, bill, system size or intent=quote).",
        )
    return repo.save_lead(None, data)


@app.get("/leads/{lead_id}", dependencies=[Depends(require_key)])
def get_lead(lead_id: int):
    lead = repo.get_lead(lead_id)
    if lead is None:
        raise HTTPException(status_code=404, detail="Lead not found")
    return {"lead": lead, "callbacks": repo.list_callbacks(lead_id), "transcript": _transcript(lead["session_id"])}


@app.patch("/leads/{lead_id}", dependencies=[Depends(require_key)])
def patch_lead(lead_id: int, fields: LeadFields):
    lead = repo.update_lead(lead_id, fields.model_dump(exclude_none=True))
    if lead is None:
        raise HTTPException(status_code=404, detail="Lead not found")
    return lead


@app.get("/stats", dependencies=[Depends(require_key)])
def stats():
    return repo.lead_stats()


@app.get("/support-requests", dependencies=[Depends(require_key)])
def support_requests(limit: int = Query(default=50, ge=1, le=200)):
    return repo.list_support_requests(limit)


# ---------- health + pages ----------

@app.get("/health")
def health():
    checks = {}
    try:
        with engine.connect() as conn:
            conn.execute(text("select 1"))
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "down"
    try:
        from rag.retriever import vector_store

        vector_store.get(limit=1)
        checks["vector_store"] = "ok"
    except Exception:
        checks["vector_store"] = "down"
    checks["llm_key"] = "configured" if os.getenv("GROQ_API_KEY") else "missing"
    ok = checks["database"] == "ok" and checks["vector_store"] == "ok" and checks["llm_key"] == "configured"
    return {"status": "ok" if ok else "degraded", **checks}


def _page(name: str):
    path = FRONTEND / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Page not built yet")
    return FileResponse(path)


@app.get("/", include_in_schema=False)
def chat_page():
    return _page("chat.html")


@app.get("/dashboard", include_in_schema=False)
def dashboard_page():
    return _page("dashboard.html")


@app.get("/whatsapp", include_in_schema=False)
def whatsapp_page():
    return _page("whatsapp.html")