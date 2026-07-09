from pydantic import BaseModel, EmailStr, Field

from luna_core.schemas.user import UserRead


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    # Also returned in the body for non-browser clients (mobile/RN) that have no
    # cookie jar; web clients ignore this and use the httpOnly refresh cookie.
    refresh_token: str | None = None
    user: UserRead


class RefreshRequest(BaseModel):
    # Mobile clients send the refresh token in the body; web omits it and the
    # httpOnly cookie is used instead.
    refresh_token: str | None = None


class VerifyEmailRequest(BaseModel):
    # The numeric code the user typed from the verification email.
    code: str = Field(min_length=4, max_length=12)


class MessageResponse(BaseModel):
    message: str
