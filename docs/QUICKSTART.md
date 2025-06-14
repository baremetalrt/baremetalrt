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
```

#### Activate the Virtual Environment (Windows)

```sh
.venv\Scripts\activate
```

#### Activate the Virtual Environment (macOS/Linux/WSL2)

```sh
source .venv/bin/activate
```

> **Note:** If you are using WSL2 (Windows Subsystem for Linux), use the Linux activation command above.
>
> **If you used a different virtual environment name (e.g., `trtllm-venv`), replace `.venv` with your venv name in all commands below.**
### Install Python Dependencies in .venv

```sh
.venv\Scripts\pip install fastapi uvicorn torch transformers
```

### Navigating to Your Project Directory (WSL2/Linux/macOS)

If your project is located on your Windows C: drive (e.g., C:\Github\mtg-something), you can access it from WSL2 with:

```sh
cd /mnt/c/Github/baremetalrt
```

Replace `mtg-something` with your actual project folder name. Use `ls /mnt/c/Github` to list available folders.

### Navigating to the Scripts Directory (WSL2/Linux/macOS)

To move into the scripts directory from your project root, use:

```sh
cd scripts
```

If you are elsewhere, use the full path (replace `USERNAME` with your Linux username):

```sh
cd /home/USERNAME/Github/baremetalrt/scripts
```

You can list available scripts with:

```sh
ls
```

### Start the Backend Server (WSL2/Linux/macOS)

1. **Navigate to your project root**

   ```sh
   cd ~/your-project-folder  # e.g., cd ~/baremetalrt or cd ~/mtg-something
   ```

2. **Activate your Python virtual environment**

   ```sh
   source .venv/bin/activate
   # Or, if your venv is named differently:
   source ~/trtllm-venv/bin/activate
   ```

3. **(Optional, but recommended) Convert scripts to Unix line endings**
   If you see errors like `$'\r': command not found`, install dos2unix and convert:
   ```sh
   sudo apt update && sudo apt install dos2unix
   dos2unix scripts/start-backend-trtllm.sh
   ```

4. **Navigate to the scripts directory**

   ```sh
   cd scripts
   ```

5. **Start the backend using the provided script**

   ```sh
   bash start-backend-trtllm.sh
   # Or, if executable:
   ./start-backend-trtllm.sh
   ```

### Start the Backend Server (Windows)

From the project root directory, run:

```sh
.\scripts\start-backend.bat
```

- This will activate the virtual environment and start the backend reliably.
- The backend will be available at: [http://localhost:8000](http://localhost:8000)
- CORS is enabled for [http://localhost:3000](http://localhost:3000) by default.
- The first startup may take several minutes while the model loads.

## 3. Adding More Models

To add a new model to your BareMetalRT setup:

1. **Add your model script** to the `scripts/` directory (e.g., `my_new_model.py`).
2. **Update the backend model list:**
   - Edit `api/model_manager.py` and add your model's ID and name to the `models` list in the `/api/models` endpoint.
3. **Update the model selection menu:**
   - Edit `scripts/start-backend.bat` and add your script name to the `MODELS` list, following the pattern of the existing entries.
   - Increment `NMODELS` accordingly.
4. **(Optional) Initialize status:**
   - Add an entry for your model in `api/model_status.json` (e.g., `"my_new_model": "offline"`).
   - The script will update its status automatically when started/stopped.

**Example:**

- Add `scripts/my_new_model.py`.
- In `api/model_manager.py`:

  ```python
  { "id": "my_new_model", "name": "My New Model" },
  ```

- In `scripts/start-backend.bat`:

  ```bat
  set MODELS[6]=my_new_model.py
  set NMODELS=7
  ```

### Public (Online) Access: Running Backend and Cloudflare Tunnel

To make your backend accessible online (for Netlify or remote frontend), you must run both your backend and the Cloudflare Tunnel in **separate terminal windows**:

**Note:**
- A *public* tunnel exposes your backend to the internet at a stable URL for anyone to access (useful for production, Netlify, or remote clients).
- A *private* tunnel restricts access to your backend (e.g., only your devices, your team, or users with Cloudflare Access/Zero Trust policies).

See below for both options.

1. **Start the backend** (from your project root):

   ```sh
   .\scripts\start-backend.bat
   ```

2. **In a new terminal window, start the Cloudflare Tunnel (production subdomain):**
   - Change directory to the `external` folder inside your project root:

     ```sh
     cd C:\Github\baremetalrt\external
     ```

   - Then start the named tunnel (replace `my-backend-tunnel` with your tunnel name if different). On Windows, use the correct executable name (as found in your `external` folder):

     ```sh
     .\cloudflared-windows-amd64.exe tunnel run my-backend-tunnel
     ```

     > **Note:** If the file is named differently (such as `cloudflared.exe`), adjust the command accordingly. The default provided with this repo is `cloudflared-windows-amd64.exe`.
     *(If your file is named differently, use the actual filename as shown in the folder.)*

   - This will expose your backend at your custom subdomain, e.g.:
     ```
     https://api.baremetalrt.ai
     ```
   - **Both windows must remain open** for your backend to be accessible online.

---

**Important: The Cloudflare Tunnel URL is now permanent and does NOT change each time you restart the tunnel!**

- Set your frontend’s API URL (in `.env.local` for local dev and as a Netlify env var for production) to your custom subdomain, e.g.:
  ```
  NEXT_PUBLIC_API_URL=https://api.baremetalrt.ai
  ```

If you do not update the API URL, your frontend will not be able to connect to the backend.

---

### Private Tunnel Access (Recommended for Personal or Restricted Use)

If you want to keep your backend private (not accessible to the public internet), you can:
- Use a tunnel with a random hostname (e.g., `cloudflared tunnel --url http://localhost:8000`), which will generate a temporary URL only you know.
- Or, use a named tunnel with Cloudflare Access/Zero Trust policies to restrict who can reach your backend (e.g., by email, IP, or login).

**To start a private tunnel (temporary/random URL):**

1. Change directory to where your cloudflared executable is located (usually `external`):
   ```sh
   cd C:\Github\baremetalrt\external
   ```
2. Run:
   ```sh
   .\cloudflared.exe tunnel --url http://localhost:8000
   ```
   *(If your file is named differently, use the actual filename as shown in the folder.)*

- This will print a unique URL (like `https://random-words.trycloudflare.com`) each time. Only those with the link can access it.
- For more security, use Cloudflare Access to require authentication for your tunnel. See [Cloudflare Zero Trust docs](https://developers.cloudflare.com/cloudflare-one/identity/) for details.

**To start a private tunnel with a permanent name but restricted access:**

1. Set up your tunnel as usual (`cloudflared tunnel create my-private-tunnel`).
2. In the Cloudflare dashboard, configure Access Policies for your tunnel subdomain.
3. Start the tunnel as you would for public, but only authorized users will be able to access it:
   ```powershell
   .\cloudflared-windows-amd64.exe tunnel run my-private-tunnel
   ```
   *(Replace `my-private-tunnel` with your actual tunnel name if it’s different. Run this from the folder containing your cloudflared executable.)*

---

**Troubleshooting:**

- If you get a "path not found" error, make sure you are running the command from your project root (e.g., `C:\Github\baremetalrt`).
- If `cloudflared-windows-amd64.exe` is not found, check the `external` folder for the correct file name and adjust the command as needed.

### Checking If Your Backend and Tunnel Are Working

After starting both the backend and the Cloudflare Tunnel, you can check if your API server and docs are accessible:

- **Local backend docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Public (tunnel) docs:** [https://picture-pockets-herald-toys.trycloudflare.com/docs](https://picture-pockets-herald-toys.trycloudflare.com/docs)

If both links load the Swagger UI, your backend and tunnel are working correctly.

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

- The frontend will be available at: [http://localhost:3000](http://localhost:3000)

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

For more details, see the main [README.md](../README.md).
