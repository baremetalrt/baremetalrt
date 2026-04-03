"""Auth routes — email/password registration and login. OAuth preserved for later."""

import hashlib

from fastapi import APIRouter, Request, Response
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel

from server.auth.jwt import create_token, create_session, revoke_user_sessions
from server.auth.middleware import get_current_user, require_auth, require_non_demo
from server import db

router = APIRouter(prefix="/auth", tags=["auth"])


# -- Helpers -----------------------------------------------------------------

def _hash_password(password: str) -> str:
    """Hash password with SHA-256. Replace with bcrypt for production scale."""
    return hashlib.sha256(password.encode()).hexdigest()


async def _issue_session(user: dict, request: Request, redirect: bool = False) -> Response:
    """Issue JWT, store session, set cookie."""
    user_id = str(user["id"])
    token, expires_at = create_token(user_id, user["email"])

    await create_session(
        user_id, token, expires_at,
        ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    if redirect:
        response = RedirectResponse(url="/app", status_code=302)
    else:
        response = JSONResponse(content={
            "ok": True,
            "user": {
                "id": user_id,
                "email": user["email"],
                "name": user.get("name"),
            },
        })

    is_https = (
        request.url.scheme == "https"
        or request.headers.get("x-forwarded-proto") == "https"
    )
    response.set_cookie(
        "token", token,
        httponly=True,
        secure=is_https,
        samesite="lax",
        max_age=60 * 60 * 72,  # 72 hours
    )
    return response


# -- Email/password auth -----------------------------------------------------

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str = ""
    first_name: str = ""
    last_name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/register")
async def register(req: RegisterRequest, request: Request):
    """Create account with email and password."""
    if not req.email or not req.password:
        return JSONResponse(status_code=400, content={"error": "Email and password required"})
    if len(req.password) < 8:
        return JSONResponse(status_code=400, content={"error": "Password must be at least 8 characters"})

    # Check if email already exists
    existing = await db.fetch_one("SELECT id FROM users WHERE email = $1", req.email.lower().strip())
    if existing:
        return JSONResponse(status_code=409, content={"error": "Email already registered"})

    # Create user
    pw_hash = _hash_password(req.password)
    first = req.first_name.strip() or req.name.split()[0] if req.name else req.email.split("@")[0]
    last = req.last_name.strip() or (req.name.split()[-1] if len(req.name.split()) > 1 else "")
    user = await db.fetch_one(
        """INSERT INTO users (email, name, first_name, last_name, password_hash, provider, provider_id)
           VALUES ($1, $2, $3, $4, $5, 'email', $1) RETURNING *""",
        req.email.lower().strip(), req.name.strip() or first, first, last, pw_hash,
    )

    return await _issue_session(dict(user), request)


@router.post("/login")
async def login(req: LoginRequest, request: Request):
    """Sign in with email and password."""
    if not req.email or not req.password:
        return JSONResponse(status_code=400, content={"error": "Email and password required"})

    user = await db.fetch_one(
        "SELECT * FROM users WHERE email = $1",
        req.email.lower().strip(),
    )
    if not user:
        return JSONResponse(status_code=401, content={"error": "Invalid email or password"})

    if user["password_hash"] != _hash_password(req.password):
        return JSONResponse(status_code=401, content={"error": "Invalid email or password"})

    # Update last login
    await db.execute("UPDATE users SET last_login = now() WHERE id = $1", user["id"])

    return await _issue_session(dict(user), request)


# -- OAuth (preserved for later) ---------------------------------------------

@router.get("/google")
async def auth_google():
    return JSONResponse(status_code=501, content={"error": "Google OAuth not configured yet"})


@router.get("/github")
async def auth_github():
    return JSONResponse(status_code=501, content={"error": "GitHub OAuth not configured yet"})


# -- Session management ------------------------------------------------------

@router.get("/me")
async def auth_me(request: Request):
    user = await get_current_user(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Not authenticated"})
    first = user.get("first_name", "")
    last = user.get("last_name", "")
    initials = ((first[:1] if first else "") + (last[:1] if last else "")).upper() or user["email"][:2].upper()
    return {
        "id": str(user["id"]),
        "email": user["email"],
        "name": user.get("name"),
        "first_name": first,
        "last_name": last,
        "avatar_url": user.get("avatar_url"),
        "initials": initials,
        "is_admin": user.get("is_admin", False),
    }


@router.put("/me")
async def update_me(request: Request):
    """Update account settings."""
    user = await require_non_demo(request)
    body = await request.json()
    updates = []
    params = []
    for field in ["first_name", "last_name", "name"]:
        if field in body:
            updates.append(f"{field} = ${len(params) + 1}")
            params.append(body[field])
    if not updates:
        return {"ok": True}
    params.append(user["id"])
    query = f"UPDATE users SET {', '.join(updates)} WHERE id = ${len(params)}"
    await db.execute(query, *params)
    return {"ok": True}


@router.post("/logout")
async def auth_logout(request: Request):
    user = await get_current_user(request)
    if user:
        await revoke_user_sessions(str(user["id"]))
    response = JSONResponse(content={"ok": True})
    response.delete_cookie("token")
    return response
