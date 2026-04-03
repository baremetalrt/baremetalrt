-- BareMetalRT schema v1
-- Run: psql -U bmrt -d baremetalrt -f 001_initial.sql

BEGIN;

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Users (from OAuth)
CREATE TABLE IF NOT EXISTS users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       TEXT UNIQUE NOT NULL,
    name        TEXT,
    avatar_url  TEXT,
    provider    TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now(),
    last_login  TIMESTAMPTZ DEFAULT now(),
    is_admin    BOOLEAN DEFAULT false,
    UNIQUE(provider, provider_id)
);

-- API keys (for daemon auth + programmatic access)
CREATE TABLE IF NOT EXISTS api_keys (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    key_hash    TEXT NOT NULL,
    key_prefix  TEXT NOT NULL,
    name        TEXT DEFAULT 'default',
    scopes      TEXT[] DEFAULT '{inference,mesh}',
    created_at  TIMESTAMPTZ DEFAULT now(),
    last_used   TIMESTAMPTZ,
    revoked     BOOLEAN DEFAULT false
);

-- Registered nodes
CREATE TABLE IF NOT EXISTS nodes (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    node_id     TEXT UNIQUE NOT NULL,
    hostname    TEXT,
    gpu_name    TEXT,
    gpu_vram_mb INTEGER,
    ip          INET,
    port        INTEGER DEFAULT 8080,
    status      TEXT DEFAULT 'offline',
    engine_name TEXT,
    last_seen   TIMESTAMPTZ DEFAULT now(),
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Sessions (JWT tracking + revocation)
CREATE TABLE IF NOT EXISTS sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    token_hash  TEXT NOT NULL,
    ip          INET,
    user_agent  TEXT,
    created_at  TIMESTAMPTZ DEFAULT now(),
    expires_at  TIMESTAMPTZ NOT NULL,
    revoked     BOOLEAN DEFAULT false
);

-- Usage tracking
CREATE TABLE IF NOT EXISTS usage (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID REFERENCES users(id),
    api_key_id        UUID REFERENCES api_keys(id),
    node_id           TEXT,
    model             TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    latency_ms        INTEGER,
    created_at        TIMESTAMPTZ DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(key_prefix);
CREATE INDEX IF NOT EXISTS idx_nodes_user ON nodes(user_id);
CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_hash ON sessions(token_hash);
CREATE INDEX IF NOT EXISTS idx_usage_user ON usage(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_created ON usage(created_at);

COMMIT;
