# Running BareMetalRT with GPU Quantization in WSL2 (Ubuntu)

This guide walks you through setting up your Windows machine with WSL2 and Ubuntu for fast, CUDA-accelerated quantized inference using bitsandbytes and HuggingFace models.

---

## 1. Install WSL2 and Ubuntu 22.04

**In PowerShell (as Administrator):**

```powershell
wsl --install -d Ubuntu-22.04
```

- Reboot if prompted.
- Create a UNIX username and password when prompted.
- Launch Ubuntu from the Start menu or by running `wsl`.

---

## 2. Update Ubuntu and Install Python

**In your Ubuntu terminal:**

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv -y
```

---

## 3. Set Up Python Virtual Environment

```bash
python3 -m venv ~/baremetalrt-venv
source ~/baremetalrt-venv/bin/activate
```

---

## 4. Install CUDA-enabled PyTorch and Dependencies

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121  # Use cu118 for CUDA 11.8 if needed
pip install bitsandbytes transformers accelerate
```

Or, if you have a requirements.txt:
```bash
pip install -r requirements.txt
```

---

## 5. Copy Your Project from Windows (Recommended)

For best performance, work from the Linux file system:

```bash
cp -r /mnt/c/Github/baremetalrt ~/
cd ~/baremetalrt
```

---

## 6. Test CUDA Availability

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

You should see `True` and your GPU name.

---

## 7. Run Your Model Scripts

```bash
python3 scripts/llama2_13b_chat_4bit.py
# or
python3 scripts/llama2_7b_chat_8int.py
```

---

## 8. Running the Backend/API

You can launch your FastAPI backend as usual:

```bash
python3 -m uvicorn api.openai_api:app --host 0.0.0.0 --port 8000
```

- Your API will be accessible from Windows at `http://localhost:8000`.

---

# Running the FastAPI GPTQ Inference Server in WSL2

## 1. Open your WSL2 Terminal
- Launch Ubuntu/WSL2 from the Start menu or Windows Terminal.

## 2. Activate your Python virtual environment (if needed)
```
source ~/baremetalrt-venv/bin/activate
```

## 3. Change to your project directory
```
cd ~/baremetalrt
```

## 4. Start the FastAPI server
```
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- The model will load (may take a minute the first time).
- The server will run at `http://localhost:8000`.

## 5. Test the API from Windows or WSL2
```
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"prompt": "What is Llama 2?"}'
```
Or use Python:
```python
import requests
response = requests.post(
    "http://localhost:8000/generate",
    json={"prompt": "What is Llama 2?"}
)
print(response.json())
```

## 6. Stopping the Server
- Press `Ctrl+C` in the WSL2 terminal to stop the server.

## Notes
- Always edit and run code in WSL2 for full GPU and package compatibility.
- For production or remote access, consider using a tunnel (e.g., Cloudflare Tunnel) or proper firewall rules.

---

## Tips
- Always run your AI scripts and backend from the Ubuntu terminal for full GPU support.
- You can edit files in Windows (with VSCode, etc.) and run them in WSL2.
- For best speed, keep your working copy in your Linux home directory (not `/mnt/c/...`).

---

**If you encounter any issues, paste the error and environment details for troubleshooting.**
