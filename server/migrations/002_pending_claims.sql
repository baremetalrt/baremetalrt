-- Pending device claims — zero-config GPU linking via IP matching
-- Run: psql -U bmrt -d baremetalrt -f 002_pending_claims.sql

BEGIN;

CREATE TABLE IF NOT EXISTS pending_claims (
    node_id     TEXT PRIMARY KEY,
    ip          INET NOT NULL,
    hostname    TEXT,
    gpu_name    TEXT,
    gpu_vram_mb INTEGER DEFAULT 0,
    claimed_by  UUID REFERENCES users(id) ON DELETE SET NULL,
    api_key_raw TEXT,              -- temporary: cleared after daemon retrieves it
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_pending_claims_ip ON pending_claims(ip);

COMMIT;
