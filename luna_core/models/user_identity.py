import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from luna_core.core.db import Base


class UserIdentity(Base):
    """An outside account (Google, Apple) a user signs in with."""

    __tablename__ = "user_identities"
    __table_args__ = (
        UniqueConstraint("provider", "subject", name="uq_user_identities_provider_subject"),
        {"schema": "core"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("core.users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # "google" | "apple"
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    # The provider's stable id for the person (the token's ``sub``).
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    # The address the provider gave, which for Apple may be a relay address
    # that differs from the account's email.
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Apple only, encrypted: what revokes the app's access when the account is
    # deleted, which Apple requires.
    provider_refresh_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
