# Contributing to BareMetalRT

Thanks for your interest in BareMetalRT. We're in early beta and welcome contributions.

## What's Open

This repo contains the **server, web UI, installer, and documentation**. The inference engine, transport layer, and daemon are in a private repo — if you're interested in contributing to those, reach out.

## Getting Started

### Server (FastAPI)

```bash
# Clone the repo
git clone https://github.com/baremetalrt/baremetalrt.git
cd baremetalrt

# Python environment
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -r requirements.txt  # or: pip install fastapi uvicorn asyncpg pyjwt

# Environment variables (see server/config.py for full list)
export DATABASE_URL="postgresql://user:pass@localhost:5432/baremetalrt"
export JWT_SECRET="your-dev-secret"
export GOOGLE_CLIENT_ID="your-google-oauth-client-id"

# Run
uvicorn server.main:app --reload --port 8080
```

### Web UI

The web UI is static HTML/JS served by the FastAPI server. Edit files in `web/` and refresh.

### Landing Site

The landing site is static HTML/CSS/JS in `site/`. It's deployed to Netlify.

## Submitting Changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Test locally — make sure the server starts and the web UI loads
4. Open a pull request with a clear description of what you changed and why

## Reporting Bugs

Open an issue using the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md). Include:

- What you expected vs. what happened
- Steps to reproduce
- GPU model, CUDA version, Windows version
- Any error messages or logs

## Security

Found a vulnerability? **Do not open a public issue.** See [SECURITY.md](SECURITY.md).

## Code Style

- Python: standard library conventions, no linter enforced yet
- JavaScript: vanilla JS, no framework, no build step
- Keep it simple. No unnecessary abstractions.

## Questions

Open a discussion or issue. We're small and responsive.
