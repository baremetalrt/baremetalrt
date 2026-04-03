-- Site-wide settings (admin-controlled key/value)
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Seed maintenance banner defaults
INSERT INTO settings (key, value) VALUES
    ('banner_1gpu', '{"enabled": true, "message": "Single GPU inference is under maintenance — back shortly."}'::jsonb),
    ('banner_2gpu', '{"enabled": true, "message": "Compute is reserved for GPU 1 — demo temporarily offline. Back shortly."}'::jsonb)
ON CONFLICT (key) DO NOTHING;
