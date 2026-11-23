import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    role: str
    is_active: bool
    is_verified: bool
    # False for an account that signs in only with Google or Apple: the
    # client hides the ways to change a password it does not have.
    has_password: bool = True
    created_at: datetime


class RoleAssign(BaseModel):
    role: str = Field(min_length=1, max_length=64)
