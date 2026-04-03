-- BareMetalRT schema v2: conversations + user memory
-- Run: psql -U bmrt -d baremetalrt -f 002_conversations.sql

BEGIN;

-- Conversations (per user, per model)
CREATE TABLE IF NOT EXISTS conversations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    title       TEXT DEFAULT 'New conversation',
    messages    JSONB DEFAULT '[]'::jsonb,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user_model ON conversations(user_id, model);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);

-- User memory (persistent context across conversations)
ALTER TABLE users ADD COLUMN IF NOT EXISTS memory TEXT DEFAULT '';

COMMIT;
