# BareMetalRT API Reference

BareMetalRT exposes an OpenAI-compatible API for chat completions.

## Authentication

All API requests require a Bearer token. Generate API keys in [Account Settings](https://baremetalrt.ai/account).

```
Authorization: Bearer bmrt_your_api_key
```

## Endpoints

### POST /v1/chat/completions

OpenAI-compatible chat completion with streaming.

```bash
curl https://baremetalrt.ai/v1/chat/completions \
  -H "Authorization: Bearer bmrt_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256,
    "stream": true
  }'
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model to use (e.g. `mistral-7b`) |
| `messages` | array | Chat messages (`role` + `content`) |
| `max_tokens` | integer | Maximum tokens to generate (default: 256) |
| `stream` | boolean | Enable SSE streaming (default: true) |

#### Response (streaming)

```
data: {"token": "Hello", "token_id": 22557, "time_ms": 78.2}
data: {"token": "!", "token_id": 28808, "time_ms": 80.1}
data: {"done": true, "total_tokens": 24}
```

### GET /api/models

List available models on connected GPU nodes.

### GET /health

Server health check. Returns node count and version.

## Client Examples

### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://baremetalrt.ai/v1",
    api_key="bmrt_your_api_key",
)

response = client.chat.completions.create(
    model="mistral-7b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=128,
)
print(response.choices[0].message.content)
```

### IDE Integration

Set the API base URL to `https://baremetalrt.ai/v1` and use your BareMetalRT API key in any OpenAI-compatible IDE plugin (Continue, Cursor, etc.).
