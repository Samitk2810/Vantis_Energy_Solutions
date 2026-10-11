from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import Base, Callback, Lead, SupportRequest, engine
from scoring.scorer import score_lead
from tools.service_area import check_service_area
from tools.validators import clean_email, clean_phone

COLUMNS = [
    "name", "phone", "email", "city", "state", "property_type", "business_type", "ownership",
    "intent", "system_size_kw", "package", "monthly_bill", "monthly_units",
    "timeline_months", "budget", "notes",
]


def init_db():
    Base.metadata.create_all(engine)


def _sanitize(fields: dict) -> dict:
    """Last line of defence: invalid phone/email are dropped, never stored."""
    out = {k: v for k, v in fields.items() if v not in (None, "", [])}
    if "phone" in out:
        out["phone"] = clean_phone(out["phone"])
    if "email" in out:
        out["email"] = clean_email(out["email"])
    return {k: v for k, v in out.items() if v not in (None, "")}


def can_persist(fields: dict) -> bool:
    """A lead is saved only with a contact method AND some requirement."""
    f = _sanitize(fields)
    has_contact = "phone" in f or "email" in f
    has_requirement = any(
        k in f for k in ("system_size_kw", "package", "monthly_bill", "monthly_units", "property_type")
    ) or f.get("intent") == "quote"
    return has_contact and has_requirement


def lead_to_dict(lead: Lead) -> dict:
    d = {c.name: getattr(lead, c.name) for c in Lead.__table__.columns}
    d["created_at"] = lead.created_at.isoformat() if lead.created_at else None
    d["updated_at"] = lead.updated_at.isoformat() if lead.updated_at else None
    return d


def _find_existing(db: Session, session_id, fields):
    if session_id:
        lead = db.scalars(select(Lead).where(Lead.session_id == session_id)).first()
        if lead:
            return lead
    if fields.get("phone"):
        lead = db.scalars(select(Lead).where(Lead.phone == fields["phone"])).first()
        if lead:
            return lead
    if fields.get("email"):
        return db.scalars(select(Lead).where(Lead.email == fields["email"])).first()
    return None


def _apply(lead: Lead, f: dict, location_status: str | None = None) -> None:
    """Write fields onto a lead, then recompute location status and score."""
    extra = dict(lead.extra or {})
    for key, value in f.items():
        if key in COLUMNS:
            setattr(lead, key, value)
        else:
            extra[key] = value
    lead.extra = extra or None
    if location_status:
        lead.location_status = location_status
    elif "city" in f:                       # city changed: recompute from the city list
        lead.location_status, _ = check_service_area(f["city"])

    merged = {c: getattr(lead, c) for c in COLUMNS}
    result = score_lead(merged, lead.location_status)
    lead.lead_score = result["lead_score"]
    lead.lead_status = result["lead_status"]
    lead.reason = result["reason"]
    lead.recommended_action = result["recommended_action"]
    lead.score_breakdown = result["score_breakdown"]


def save_lead(session_id: str | None, fields: dict, location_status: str | None = None) -> dict:
    """Create or update a lead (matched by session, phone or email), then re-score it."""
    f = _sanitize(fields)
    with Session(engine) as db:
        lead = _find_existing(db, session_id, f)
        if lead is None:
            lead = Lead(session_id=session_id)
            db.add(lead)
        elif session_id and not lead.session_id:
            lead.session_id = session_id
        _apply(lead, f, location_status)
        db.commit()
        db.refresh(lead)
        return lead_to_dict(lead)


def update_lead(lead_id: int, fields: dict) -> dict | None:
    """Update a lead by id (used by the dashboard), then re-score it."""
    f = _sanitize(fields)
    with Session(engine) as db:
        lead = db.get(Lead, lead_id)
        if lead is None:
            return None
        _apply(lead, f)
        db.commit()
        db.refresh(lead)
        return lead_to_dict(lead)


def get_lead(lead_id: int) -> dict | None:
    with Session(engine) as db:
        lead = db.get(Lead, lead_id)
        return lead_to_dict(lead) if lead else None


def list_leads(status: str | None = None, city: str | None = None, limit: int = 50, offset: int = 0):
    with Session(engine) as db:
        q = select(Lead).order_by(Lead.lead_score.desc(), Lead.created_at.desc())
        if status:
            q = q.where(Lead.lead_status == status.upper())
        if city:
            q = q.where(Lead.city.ilike(city))
        return [lead_to_dict(x) for x in db.scalars(q.limit(limit).offset(offset))]


def create_callback(lead_id: int, preferred_time: str | None = None, note: str | None = None) -> dict:
    with Session(engine) as db:
        cb = Callback(lead_id=lead_id, preferred_time=preferred_time, note=note)
        db.add(cb)
        db.commit()
        db.refresh(cb)
        return {"id": cb.id, "lead_id": lead_id, "preferred_time": cb.preferred_time,
                "note": cb.note, "status": cb.status}


def create_support_request(session_id, name, contact, project, issue) -> dict:
    with Session(engine) as db:
        req = SupportRequest(session_id=session_id, name=name, contact=contact,
                             project=project, issue=issue)
        db.add(req)
        db.commit()
        db.refresh(req)
        return {"id": req.id, "name": name, "contact": contact, "project": project,
                "issue": issue, "status": req.status}

def list_callbacks(lead_id: int) -> list[dict]:
    with Session(engine) as db:
        rows = db.scalars(select(Callback).where(Callback.lead_id == lead_id).order_by(Callback.id.desc()))
        return [{"id": c.id, "preferred_time": c.preferred_time, "note": c.note, "status": c.status,
                 "created_at": c.created_at.isoformat()} for c in rows]


def list_support_requests(limit: int = 50) -> list[dict]:
    with Session(engine) as db:
        rows = db.scalars(select(SupportRequest).order_by(SupportRequest.id.desc()).limit(limit))
        return [{"id": r.id, "name": r.name, "contact": r.contact, "project": r.project,
                 "issue": r.issue, "status": r.status, "created_at": r.created_at.isoformat()}
                for r in rows]


def lead_stats() -> dict:
    with Session(engine) as db:
        by_status = {"HOT": 0, "WARM": 0, "COLD": 0}
        for status, count in db.execute(select(Lead.lead_status, func.count()).group_by(Lead.lead_status)):
            by_status[status] = count
        by_city = [
            {"city": city or "Unknown", "count": count}
            for city, count in db.execute(
                select(Lead.city, func.count()).group_by(Lead.city).order_by(func.count().desc()).limit(8)
            )
        ]
        avg = db.scalar(select(func.avg(Lead.lead_score)))
        return {
            "total_leads": sum(by_status.values()),
            "by_status": by_status,
            "by_city": by_city,
            "avg_score": round(avg, 1) if avg is not None else 0,
            "callbacks_requested": db.scalar(select(func.count(Callback.id)).where(Callback.status == "requested")) or 0,
            "open_support_requests": db.scalar(select(func.count(SupportRequest.id)).where(SupportRequest.status == "open")) or 0,
        }