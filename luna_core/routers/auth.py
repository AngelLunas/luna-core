import logging

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Cookie,
    HTTPException,
    Request,
    Response,
    status,
)

from luna_core.core.config import settings
from luna_core.core.dependencies import (
    CurrentUserAllowUnverified,
    DBSession,
    RedisClient,
    get_client_ip,
)
from luna_core.core.rate_limit import check_rate_limit, reset_rate_limit
from luna_core.models.user import User
from luna_core.schemas.auth import (
    LoginRequest,
    MessageResponse,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    VerifyEmailRequest,
)
from luna_core.schemas.user import UserRead
from luna_core.services.auth import (
    IssuedTokens,
    authenticate_user,
    issue_tokens,
    register_user,
    revoke_refresh_token,
    rotate_refresh_token,
)
from luna_core.services.verification import create_verification_code, verify_code

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# verify_code reasons that are the user's fault → 400 with a stable code the
# client maps to localized copy.
_VERIFY_ERROR_STATUS = {
    "invalid": status.HTTP_400_BAD_REQUEST,
    "expired": status.HTTP_400_BAD_REQUEST,
    "no_code": status.HTTP_400_BAD_REQUEST,
    "too_many_attempts": status.HTTP_429_TOO_MANY_REQUESTS,
}


def _primary_language(accept_language: str | None) -> str | None:
    """The primary language subtag from an Accept-Language header.

    "es-419,es;q=0.9,en;q=0.8" -> "es". Returns None when absent so the host
    sender falls back to its own default.
    """
    if not accept_language:
        return None
    first = accept_language.split(",")[0].strip()
    code = first.split(";")[0].split("-")[0].strip().lower()
    return code or None


async def _dispatch_verification_email(
    request: Request, db: DBSession, background: BackgroundTasks, user: User
) -> None:
    """Mint a verification code and hand it to the host app's e-mail hook.

    No-op unless verification is required AND the host registered a sender on
    ``app.state.send_verification_email`` (signature:
    ``async (email, code, *, lang)``). ``lang`` comes from the client's
    Accept-Language so the email matches the UI language. Sending runs in a
    background task so the request stays snappy.
    """
    if not settings.email_verification_required:
        return
    sender = getattr(request.app.state, "send_verification_email", None)
    if sender is None:
        logger.warning("email_verification_required but no send_verification_email hook")
        return
    lang = _primary_language(request.headers.get("accept-language"))
    raw_code = await create_verification_code(db, user)
    background.add_task(sender, user.email, raw_code, lang=lang)


def _set_refresh_cookie(response: Response, tokens: IssuedTokens) -> None:
    max_age = settings.refresh_token_expire_days * 24 * 60 * 60
    response.set_cookie(
        key=settings.refresh_cookie_name,
        value=tokens.refresh_token,
        max_age=max_age,
        expires=max_age,
        path=settings.api_v1_prefix + "/auth",
        domain=settings.refresh_cookie_domain,
        secure=settings.refresh_cookie_secure,
        httponly=True,
        samesite=settings.refresh_cookie_samesite,
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(
        key=settings.refresh_cookie_name,
        path=settings.api_v1_prefix + "/auth",
        domain=settings.refresh_cookie_domain,
    )


def _token_response(tokens: IssuedTokens) -> TokenResponse:
    return TokenResponse(
        access_token=tokens.access_token,
        token_type="bearer",
        expires_in=tokens.access_expires_in,
        refresh_token=tokens.refresh_token,
        user=UserRead.model_validate(tokens.user),
    )


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    payload: RegisterRequest,
    request: Request,
    response: Response,
    background: BackgroundTasks,
    db: DBSession,
) -> TokenResponse:
    user = await register_user(db, payload.email, payload.password)
    # Tokens are issued even when unverified: the client lands on the "confirm
    # your email" screen with a live session (it can poll /me and log out). The
    # verification gate keeps every other protected route 403 until verified.
    tokens = await issue_tokens(db, user)
    await _dispatch_verification_email(request, db, background, user)
    _set_refresh_cookie(response, tokens)
    return _token_response(tokens)


@router.post("/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: DBSession,
    redis: RedisClient,
) -> TokenResponse:
    ip = get_client_ip(request)
    rate_key = f"rl:login:{ip}"
    rate = await check_rate_limit(
        redis,
        rate_key,
        limit=settings.login_rate_limit_attempts,
        window_seconds=settings.login_rate_limit_window_seconds,
    )
    if not rate.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please try again later.",
            headers={"Retry-After": str(rate.retry_after)},
        )

    user = await authenticate_user(db, payload.email, payload.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    await reset_rate_limit(redis, rate_key)

    tokens = await issue_tokens(db, user)
    _set_refresh_cookie(response, tokens)
    return _token_response(tokens)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    response: Response,
    db: DBSession,
    body: RefreshRequest | None = None,
    cookie_token: str | None = Cookie(
        default=None, alias=settings.refresh_cookie_name
    ),
) -> TokenResponse:
    # Mobile sends the token in the body; web relies on the httpOnly cookie.
    refresh_token = (body.refresh_token if body else None) or cookie_token
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing refresh token",
        )
    tokens = await rotate_refresh_token(db, refresh_token)
    _set_refresh_cookie(response, tokens)
    return _token_response(tokens)


@router.post("/logout", response_model=MessageResponse)
async def logout(
    response: Response,
    db: DBSession,
    body: RefreshRequest | None = None,
    cookie_token: str | None = Cookie(
        default=None, alias=settings.refresh_cookie_name
    ),
) -> MessageResponse:
    refresh_token = (body.refresh_token if body else None) or cookie_token
    if refresh_token:
        await revoke_refresh_token(db, refresh_token)
    _clear_refresh_cookie(response)
    return MessageResponse(message="Logged out")


@router.get("/me", response_model=UserRead)
async def me(current_user: CurrentUserAllowUnverified) -> UserRead:
    # Allow-unverified: the client reads ``is_verified`` here to drive the
    # "confirm your email" screen, so this must answer before verification.
    return UserRead.model_validate(current_user)


@router.post("/verify-email", response_model=UserRead)
async def verify_email(
    payload: VerifyEmailRequest,
    db: DBSession,
    redis: RedisClient,
    current_user: CurrentUserAllowUnverified,
) -> UserRead:
    """The app posts the code the user typed from the email. On a match the
    account flips to verified and the (now verified) user is returned."""
    # Defence-in-depth on top of the per-code attempt cap: throttle total guesses.
    rate = await check_rate_limit(
        redis, f"rl:verify-email:{current_user.id}", limit=15, window_seconds=600
    )
    if not rate.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many attempts. Please wait and request a new code.",
            headers={"Retry-After": str(rate.retry_after)},
        )

    result = await verify_code(db, current_user, payload.code)
    if not result.ok:
        raise HTTPException(
            status_code=_VERIFY_ERROR_STATUS.get(result.reason, 400),
            detail=result.reason,
        )
    return UserRead.model_validate(current_user)


@router.post("/resend-verification", response_model=MessageResponse)
async def resend_verification(
    request: Request,
    background: BackgroundTasks,
    db: DBSession,
    redis: RedisClient,
    current_user: CurrentUserAllowUnverified,
) -> MessageResponse:
    if current_user.is_verified:
        return MessageResponse(message="Already verified")

    # Light throttle so the button can't be hammered into a mail flood.
    rate = await check_rate_limit(
        redis, f"rl:verify-resend:{current_user.id}", limit=4, window_seconds=600
    )
    if not rate.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please wait before resending.",
            headers={"Retry-After": str(rate.retry_after)},
        )

    await _dispatch_verification_email(request, db, background, current_user)
    return MessageResponse(message="Verification email sent")
