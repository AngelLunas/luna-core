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


class GoogleSignInRequest(BaseModel):
    id_token: str = Field(min_length=1, max_length=8192)


class AppleSignInRequest(BaseModel):
    identity_token: str = Field(min_length=1, max_length=8192)
    # One-time code that exchanges for the refresh token we revoke when the
    # account is deleted (Apple requires it).
    authorization_code: str | None = Field(default=None, max_length=1024)
    # The nonce the app asked Apple to sign into this token.
    nonce: str | None = Field(default=None, max_length=256)


class AppleNotification(BaseModel):
    # Apple's server-to-server notification: a JWT it signs.
    payload: str = Field(min_length=1, max_length=16384)


class RefreshRequest(BaseModel):
    # Mobile clients send the refresh token in the body; web omits it and the
    # httpOnly cookie is used instead.
    refresh_token: str | None = None


class VerifyEmailRequest(BaseModel):
    # The numeric code the user typed from the verification email.
    code: str = Field(min_length=4, max_length=12)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    code: str = Field(min_length=4, max_length=12)
    new_password: str = Field(min_length=8, max_length=128)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class MessageResponse(BaseModel):
    message: str
