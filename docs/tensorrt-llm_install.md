# TensorRT-LLM Out-of-the-Box Installation (WSL2)

This guide describes how to set up TensorRT-LLM for inference and engine export using the official NVIDIA pip wheel, with no source build required. These steps ensure maximum compatibility and minimize build issues.

---

## 1. Prerequisites
- **WSL2** with Ubuntu 20.04/22.04
- **NVIDIA GPU** with drivers and CUDA 12.1 or 12.4 installed
- **Python 3.10** (recommended)

---

## 2. Create a Clean Python Virtual Environment
```bash
python3.10 -m venv ~/trtllm-venv
source ~/trtllm-venv/bin/activate
```

---

## 3. Install PyTorch (CUDA 12.1)
```bash
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

---

## 4. Install Required Dependencies
```bash
pip install nvidia-modelopt[torch]==0.27.0
pip install tensorrt==10.9.0
pip install transformers==4.51.0
```

---

## 5. Install TensorRT-LLM (Official NVIDIA Wheel)
```bash
pip install tensorrt-llm==0.19.0 --extra-index-url https://pypi.nvidia.com
```
*(If 0.19.0 is not available, omit the version to get the latest compatible one.)*

---

## 6. Verify Installation
```bash
python -c "import torch; print(torch.__version__)"
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
python -c 'import transformers; print(transformers.__version__)'
python -c 'import tensorrt; print(tensorrt.__version__)'
```
All versions should match the requirements above.

---

## 7. (Optional) Troubleshooting
- If you see `undefined symbol` errors, ensure all versions match exactly.
- Use a clean venv and do not mix with bitsandbytes or other conflicting packages.
- If you need to change CUDA/PyTorch versions, recreate the venv and reinstall all packages.

---

## 8. Next Steps
- You can now use TensorRT-LLM for engine export, quantization, and inference without a source build.
- For advanced features or custom ops, a source build may still be required.

---

**This workflow has been tested and confirmed to work on WSL2 with the versions above.**

If you encounter any issues, check your package versions and consult the official [NVIDIA TensorRT-LLM documentation](https://nvidia.github.io/TensorRT-LLM/installation/linux.html).
