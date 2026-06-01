from fastapi import APIRouter, Cookie, HTTPException, Request, Response, status

from luna_core.core.config import settings
from luna_core.core.dependencies import CurrentUser, DBSession, RedisClient, get_client_ip
from luna_core.core.rate_limit import check_rate_limit, reset_rate_limit
from luna_core.schemas.auth import (
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    TokenResponse,
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

router = APIRouter(prefix="/auth", tags=["auth"])


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
        user=UserRead.model_validate(tokens.user),
    )


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    payload: RegisterRequest,
    response: Response,
    db: DBSession,
) -> TokenResponse:
    user = await register_user(db, payload.email, payload.password)
    tokens = await issue_tokens(db, user)
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
    refresh_token: str | None = Cookie(default=None, alias=settings.refresh_cookie_name),
) -> TokenResponse:
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
    refresh_token: str | None = Cookie(default=None, alias=settings.refresh_cookie_name),
) -> MessageResponse:
    if refresh_token:
        await revoke_refresh_token(db, refresh_token)
    _clear_refresh_cookie(response)
    return MessageResponse(message="Logged out")


@router.get("/me", response_model=UserRead)
async def me(current_user: CurrentUser) -> UserRead:
    return UserRead.model_validate(current_user)
