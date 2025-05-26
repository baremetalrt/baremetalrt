# BareMetalRT Quick Start

This guide provides the essential commands and steps to launch the backend and frontend for local development.

## 1. Prerequisites
- Python 3.8+
- Node.js (for frontend)
- Required Python packages: `fastapi`, `uvicorn`, `torch`, `transformers`

## 2. Backend (API) Setup

### Create and Activate Virtual Environment
```sh
python -m venv .venv
.venv\Scripts\activate
```

### Install Python Dependencies in .venv
```sh
.venv\Scripts\pip install fastapi uvicorn torch transformers
```

### Start the Backend Server
From the project root directory, run:
```sh
.\scripts\start-backend.bat
```
- This will activate the virtual environment and start the backend reliably.
- The backend will be available at: http://localhost:8000
- CORS is enabled for http://localhost:3000 by default.
- The first startup may take several minutes while the model loads.

### Public (Online) Access: Running Backend and Cloudflare Tunnel
To make your backend accessible online (for Netlify or remote frontend), you must run both your backend and the Cloudflare Tunnel in **separate terminal windows**:

1. **Start the backend** (as above):
   ```sh
   .\scripts\start-backend.bat
   ```
2. **In a new terminal window, start the Cloudflare Tunnel:**
   ```sh
   cd external
   cloudflared\cloudflared-windows-amd64.exe tunnel --url http://localhost:8000
   ```
   - This will provide a public `.trycloudflare.com` URL for your backend.
   - Both windows must remain open for your backend to be accessible online.

#### Troubleshooting
- If you see `ModuleNotFoundError`, make sure you installed dependencies using `.venv\Scripts\pip` and are running the backend from the project root.
- If `/docs` is not available, wait for the model to finish loading (watch the terminal for readiness messages).
- For port conflicts, ensure no other process is using 8000 or 3000.

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
