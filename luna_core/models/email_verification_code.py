import uuid
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from luna_core.core.db import Base

if TYPE_CHECKING:
    from luna_core.models.user import User


class CodePurpose(StrEnum):
    VERIFY_EMAIL = "verify_email"
    PASSWORD_RESET = "password_reset"


class EmailVerificationCode(Base):
    """A short, single-use numeric code e-mailed to confirm address ownership.

    Only the SHA-256 hash of the code is stored. Brute force is bounded by
    ``attempts`` (the row is burned once the cap is hit) plus a short expiry —
    not by the (low) entropy of a 6-digit code. Minting a fresh code marks the
    user's prior unused codes spent, so exactly one is ever live.
    """

    __tablename__ = "email_verification_codes"
    __table_args__ = {"schema": "core"}

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("core.users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # What typing the code proves: owning the address, or the right to set a
    # new password. Each purpose keeps its own single live code.
    purpose: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=CodePurpose.VERIFY_EMAIL,
        server_default=CodePurpose.VERIFY_EMAIL.value,
    )
    code_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship()
