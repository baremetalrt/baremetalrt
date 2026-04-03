"""OAuth handlers for Google and GitHub."""

import httpx

from server.config import (
    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET,
    GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET,
    BASE_URL,
)


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


def google_auth_url() -> str:
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": f"{BASE_URL}/auth/google/callback",
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{GOOGLE_AUTH_URL}?{qs}"


async def google_exchange(code: str) -> dict:
    """Exchange authorization code for user info."""
    async with httpx.AsyncClient() as client:
        # Exchange code for token
        resp = await client.post(GOOGLE_TOKEN_URL, data={
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": f"{BASE_URL}/auth/google/callback",
        })
        resp.raise_for_status()
        tokens = resp.json()

        # Get user info
        resp = await client.get(GOOGLE_USERINFO_URL, headers={
            "Authorization": f"Bearer {tokens['access_token']}",
        })
        resp.raise_for_status()
        info = resp.json()

    return {
        "provider": "google",
        "provider_id": info["id"],
        "email": info["email"],
        "name": info.get("name"),
        "avatar_url": info.get("picture"),
    }


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------

GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_EMAILS_URL = "https://api.github.com/user/emails"


def github_auth_url() -> str:
    params = {
        "client_id": GITHUB_CLIENT_ID,
        "redirect_uri": f"{BASE_URL}/auth/github/callback",
        "scope": "read:user user:email",
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{GITHUB_AUTH_URL}?{qs}"


async def github_exchange(code: str) -> dict:
    """Exchange authorization code for user info."""
    async with httpx.AsyncClient() as client:
        # Exchange code for token
        resp = await client.post(GITHUB_TOKEN_URL, data={
            "client_id": GITHUB_CLIENT_ID,
            "client_secret": GITHUB_CLIENT_SECRET,
            "code": code,
        }, headers={"Accept": "application/json"})
        resp.raise_for_status()
        tokens = resp.json()
        access_token = tokens["access_token"]

        # Get user info
        resp = await client.get(GITHUB_USER_URL, headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        })
        resp.raise_for_status()
        user = resp.json()

        # Get primary email (may be private)
        email = user.get("email")
        if not email:
            resp = await client.get(GITHUB_EMAILS_URL, headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            })
            resp.raise_for_status()
            emails = resp.json()
            primary = next((e for e in emails if e["primary"]), emails[0] if emails else None)
            email = primary["email"] if primary else f"{user['id']}@github.noreply"

    return {
        "provider": "github",
        "provider_id": str(user["id"]),
        "email": email,
        "name": user.get("name") or user.get("login"),
        "avatar_url": user.get("avatar_url"),
    }
