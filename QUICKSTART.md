# BareMetalRT Quick Start

This guide provides the essential commands and steps to launch the backend and frontend for local development.

## 1. Prerequisites
- Python 3.8+
- Node.js (for frontend)
- Required Python packages: `fastapi`, `uvicorn`, `torch`, `transformers`

## 2. Backend (API) Setup

### Install Python Dependencies
```sh
pip install fastapi uvicorn torch transformers
```

### Start the Backend Server
```sh
uvicorn api.openai_api:app --host 0.0.0.0 --port 8000
```

- The backend will be available at: http://localhost:8000
- CORS is enabled for http://localhost:3000 by default.

### Stopping the Backend Server (Windows)
```sh
taskkill /F /IM uvicorn.exe
```

---

## 3. Frontend (Chat UI) Setup

### Install Frontend Dependencies
```sh
cd chat-ui
npm install
```

### Start the Frontend Dev Server
```sh
npm run dev
```

- The frontend will be available at: http://localhost:3000

### Stop the Frontend Dev Server
- In the terminal running `npm run dev`, press `Ctrl+C` to stop the server.

### Restart the Frontend Dev Server
- From the `chat-ui` directory:
  ```sh
  npm run dev
  ```

---

## 4. Troubleshooting
- If you see CORS errors, ensure the backend is running and CORS is enabled in `api/openai_api.py`.
- If you change backend code, restart the backend server.
- For port conflicts, ensure no other process is using 8000 or 3000.

---

## 5. Useful Commands
- **Restart backend:**
  - Stop: `taskkill /F /IM uvicorn.exe`
  - Start: `uvicorn api.openai_api:app --host 0.0.0.0 --port 8000`
- **Restart frontend:**
  - Stop: Ctrl+C in terminal
  - Start: `npm run dev` (from `chat-ui` directory)

---

For more details, see the main `README.md`.
