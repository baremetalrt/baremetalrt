# OpenAI-Compatible API Server for Local Llama 2 Inference

## 🚀 Quick Start

1. **Install dependencies:**
   ```sh
   pip install fastapi uvicorn torch transformers
   ```

2. **Start the API server (recommended):**
   ```sh
   python -m uvicorn api.openai_api:app --host 0.0.0.0 --port 8000
   ```
   This method works on all platforms and does not require uvicorn to be in your PATH.

3. **Test the API:**
   - Use the provided `scripts/test_api.py` script:
     ```sh
     python scripts/test_api.py
     ```
   - Or use Postman/curl to POST to `http://localhost:8000/v1/completions`.

---

This document describes how to install dependencies, start the FastAPI server, and test the OpenAI-compatible API for local single-node inference.

## Requirements
- Python 3.10+
- Windows (tested)
- GPU recommended for best performance

## Install Dependencies
```
pip install fastapi uvicorn torch transformers
```

## Start the API Server
Recommended (works everywhere):
```
python -m uvicorn api.openai_api:app --host 0.0.0.0 --port 8000
```

## API Endpoint
- **POST** `/v1/completions`  
  OpenAI-compatible completion endpoint. Accepts JSON payload with `prompt`, `max_tokens`, etc.

## Example API Test
Run the following script to test the API after starting the server:

```
python test_api.py
```

This will send a sample prompt and print the response. You should see a JSON object with `choices`, `id`, `object`, etc., matching the OpenAI API format.

---

For troubleshooting, ensure your model and dependencies are installed, and check the console for errors during server startup.
