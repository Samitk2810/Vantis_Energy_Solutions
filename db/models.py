import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/vantis.db")
if DATABASE_URL.startswith("sqlite:///"):
    Path(DATABASE_URL.replace("sqlite:///", "", 1)).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


def _now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Lead(Base):
    __tablename__ = "leads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True)
    name: Mapped[str | None] = mapped_column(String(120))
    phone: Mapped[str | None] = mapped_column(String(20), index=True)
    email: Mapped[str | None] = mapped_column(String(120), index=True)
    city: Mapped[str | None] = mapped_column(String(80))
    state: Mapped[str | None] = mapped_column(String(80))
    property_type: Mapped[str | None] = mapped_column(String(40))
    business_type: Mapped[str | None] = mapped_column(String(80))
    ownership: Mapped[str | None] = mapped_column(String(40))
    intent: Mapped[str | None] = mapped_column(String(20))
    system_size_kw: Mapped[float | None] = mapped_column(Float)
    package: Mapped[str | None] = mapped_column(String(80))
    monthly_bill: Mapped[float | None] = mapped_column(Float)
    monthly_units: Mapped[float | None] = mapped_column(Float)
    timeline_months: Mapped[float | None] = mapped_column(Float)
    budget: Mapped[str | None] = mapped_column(String(80))
    notes: Mapped[str | None] = mapped_column(Text)
    extra: Mapped[dict | None] = mapped_column(JSON)
    location_status: Mapped[str | None] = mapped_column(String(20))
    lead_score: Mapped[int] = mapped_column(Integer, default=0)
    lead_status: Mapped[str] = mapped_column(String(10), default="COLD", index=True)
    reason: Mapped[str | None] = mapped_column(Text)
    recommended_action: Mapped[str | None] = mapped_column(Text)
    score_breakdown: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class Callback(Base):
    __tablename__ = "callbacks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    lead_id: Mapped[int] = mapped_column(ForeignKey("leads.id"), index=True)
    preferred_time: Mapped[str | None] = mapped_column(String(120))
    note: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="requested")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SupportRequest(Base):
    __tablename__ = "support_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True)
    name: Mapped[str | None] = mapped_column(String(120))
    contact: Mapped[str | None] = mapped_column(String(120))
    project: Mapped[str | None] = mapped_column(Text)
    issue: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)