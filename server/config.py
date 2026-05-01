"""BareMetalRT server configuration. All secrets from environment variables."""

import os

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# OAuth — Google
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")

# OAuth — GitHub
GITHUB_CLIENT_ID = os.environ.get("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "")

# JWT
JWT_SECRET = os.environ.get("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 72

# Server
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8080")

# Public demo: anonymous /demo traffic mirrors this user's GPUs.
# Empty disables the public demo path (falls back to "any daemon").
PUBLIC_DEMO_USER_ID = os.environ.get("PUBLIC_DEMO_USER_ID", "")
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://baremetalrt.com,https://www.baremetalrt.com,http://localhost:8080",
).split(",")
