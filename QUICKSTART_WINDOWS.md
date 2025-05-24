# Petals Quickstart Guide for Windows

This guide helps you get Petals running natively on Windows—no WSL or Docker required!

---

## 1. Prerequisites

- **Windows 10/11** (x64)
- **Python 3.10+** (from https://python.org)
- **Git** (from https://git-scm.com)
- **Go** (for building `p2pd.exe`, from https://go.dev)
- **A CUDA-capable GPU** (NVIDIA, with up-to-date drivers)

---

## 2. Clone the Repo & Prepare Environment

```powershell
git clone <your-fork-or-main-repo-url>
cd <repo-directory>
python -m venv venv
venv\Scripts\activate
```

---

## 3. Build and Place `p2pd.exe`

1. Download and install Go from https://go.dev.
2. Build the binary:
   ```powershell
   git clone https://github.com/libp2p/go-libp2p-daemon.git
   cd go-libp2p-daemon
   go build -o p2pd.exe ./p2pd
   # Copy p2pd.exe to your project root or the required location
   ```

---

## 4. Install Dependencies (Patched Hivemind & Petals)

```powershell
pip install --upgrade pip
pip uninstall hivemind uvloop -y
pip install -e ./external/hivemind
pip install --no-deps git+https://github.com/bigscience-workshop/petals
pip install dijkstar
```

---

## 5. HuggingFace Login (for Model Access)

```powershell
pip install huggingface_hub
huggingface-cli login
```
- Request access to Llama weights if needed: https://huggingface.co/meta-llama

---

## 6. Run a Petals Node (Join the Mesh)

```powershell
python -m petals.cli.run_server meta-llama/Llama-2-70b-hf
```
- Replace with the model you have access to (e.g. Llama-3).
- You should see logs about connecting to the mesh and serving blocks.

---

## 7. Run Distributed Inference (Client Example)

```python
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

model_name = "meta-llama/Llama-2-70b-hf"  # Or the model you used above
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoDistributedModelForCausalLM.from_pretrained(model_name)

inputs = tokenizer("Hello, world!", return_tensors="pt")["input_ids"]
outputs = model.generate(inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```

---

## 8. Troubleshooting

- **Missing dependencies:** Install with `pip install <package>` as needed.
- **Protobuf import errors:** Ensure you are using the patched Hivemind from `external/hivemind`.
- **`p2pd.exe` not found:** Double-check its location.
- **For more details:** See `windows_patches.md`.

---

## 9. Resources
- [Petals Documentation](https://petals.dev/)
- [HuggingFace Model Access](https://huggingface.co/meta-llama)
- [windows_patches.md](./windows_patches.md)

---

**You are now running Petals natively on Windows and contributing to the global open-source AI mesh!**
