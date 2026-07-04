"""SQLAlchemy ORM models for VoiceAgent.

Tables: Business, User, FAQ, Call, Appointment.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Business(Base):
    """A business that subscribes to the VoiceAgent service.

    This is the core multi-tenant entity. All other data is scoped
    to a business.
    """

    __tablename__ = "businesses"

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    phone_number = Column(String(20), unique=True, nullable=False)
    greeting_message = Column(
        Text,
        default="Hello, you've reached {business_name}. How can I help you today?",
    )
    business_hours = Column(JSON, default=dict)
    timezone = Column(String(50), default="America/New_York")
    twilio_phone_sid = Column(String(100), default="")
    twilio_auth_token = Column(Text, default="")
    google_calendar_encrypted = Column(Text, default="")
    google_calendar_id = Column(String(200), default="primary")
    transfer_phone_number = Column(String(20), default="")
    webhook_secret = Column(String(100), default="")
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship(
        "User", back_populates="business", uselist=False, cascade="all, delete-orphan"
    )
    faqs = relationship("FAQ", back_populates="business", cascade="all, delete-orphan")
    calls = relationship(
        "Call", back_populates="business", cascade="all, delete-orphan"
    )
    appointments = relationship(
        "Appointment", back_populates="business", cascade="all, delete-orphan"
    )


class User(Base):
    """Dashboard login credential. One per business for MVP."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    business_id = Column(
        Integer, ForeignKey("businesses.id"), unique=True, nullable=False
    )
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    business = relationship("Business", back_populates="user")


class FAQ(Base):
    """Knowledge base entry. The AI searches these to answer caller questions."""

    __tablename__ = "faqs"

    id = Column(Integer, primary_key=True)
    business_id = Column(Integer, ForeignKey("businesses.id"), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    category = Column(String(100), default="general")
    keywords = Column(Text, default="")
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    business = relationship("Business", back_populates="faqs")


class Call(Base):
    """Record of every inbound call."""

    __tablename__ = "calls"

    id = Column(Integer, primary_key=True)
    business_id = Column(Integer, ForeignKey("businesses.id"), nullable=False)
    caller_number = Column(String(20), nullable=False)
    duration_seconds = Column(Integer, default=0)
    transcript_json = Column(JSON, default=list)
    outcome = Column(
        Enum(
            "appointment_booked",
            "faq_answered",
            "transferred",
            "hangup",
            "error",
            name="call_outcome",
        ),
        default="hangup",
    )
    twilio_call_sid = Column(String(100), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    business = relationship("Business", back_populates="calls")
    appointments = relationship("Appointment", back_populates="call")


class Appointment(Base):
    """Appointment booked by the AI through the call flow."""

    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True)
    call_id = Column(Integer, ForeignKey("calls.id"), nullable=True)
    business_id = Column(Integer, ForeignKey("businesses.id"), nullable=False)
    customer_name = Column(String(200), nullable=False)
    customer_phone = Column(String(20), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    description = Column(Text, default="")
    status = Column(
        Enum(
            "confirmed",
            "cancelled",
            "no_show",
            name="appointment_status",
        ),
        default="confirmed",
    )
    google_event_id = Column(String(200), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    business = relationship("Business", back_populates="appointments")
    call = relationship("Call", back_populates="appointments")
