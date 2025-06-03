<p align="center">
    <img src="https://i.imgur.com/7eR7Pan.png" width="400"><br>
    Run large language models at home, BitTorrent-style.<br>
    Fine-tuning and inference <a href="https://github.com/bigscience-workshop/petals#benchmarks">up to 10x faster</a> than offloading
    <br><br>
    <a href="https://pypi.org/project/petals/"><img src="https://img.shields.io/pypi/v/petals.svg?color=green"></a>
    <a href="https://discord.gg/tfHfe8B34k"><img src="https://img.shields.io/discord/865254854262652969?label=discord&logo=discord&logoColor=white"></a>
    <br>
</p>

---

## 🚀 Windows Compatibility (Native, No WSL Required!)

**Petals now runs natively on Windows!**

- All core features are supported, including distributed inference and DHT node hosting.
- See [`windows_patches.md`](./windows_patches.md) for a full list of patches, workarounds, and step-by-step instructions.
- Major changes:
  - Patched/forked Hivemind for Windows (no `uvloop` required)
  - Manual build and placement of the `p2pd.exe` binary (see below)
  - All protobuf imports fixed for Windows Python packaging
- **Limitations:**
  - No `uvloop` (uses default asyncio event loop)
  - Docker and WSL are not required for basic usage, but some advanced features may still require a Linux environment.
  - See `windows_patches.md` for any known issues or limitations.

### Quickstart

### 1. Launch the Backend

```bash
# Activate your Python virtual environment
source ~/trtllm-venv/bin/activate

# Change to your project directory
cd /mnt/c/Github/baremetalrt

# Start the backend (model selection prompt will appear)
bash scripts/start-backend-trtllm.sh
```

### 2. Start Cloudflared Tunnel

```bash
# In a new terminal (while backend is running)
cloudflared tunnel run <YOUR_TUNNEL_NAME>
```
- Replace `<YOUR_TUNNEL_NAME>` with your configured tunnel name (see your Cloudflare dashboard or tunnel config).
- Make sure your Cloudflared tunnel is configured to point to `localhost:8000` (or your backend port).

### 3. Troubleshooting
- If you get `No such file or directory`, make sure you are in the correct project directory.
- If you see long delays on first request, wait for the backend model to finish warming up. You can check readiness at `/api/health` (returns `{"status": "ready"}` when ready).
- If you get 503 errors, the model is still warming up—try again in a few seconds.
- For Cloudflared setup help, see https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/tunnel-guide/ for Windows

1. **Install Python 3.8+ and create a virtual environment**
2. **Install Go (for building p2pd.exe):** https://go.dev/dl/
3. **Clone this repo and the forked Hivemind submodule**
4. **Build `p2pd.exe` from source and place it in the expected directory**
5. **Install dependencies (see below), then run Petals as usual!**

For full details and troubleshooting, see [`windows_patches.md`](./windows_patches.md).

Generate text with distributed **Llama 3.1** (up to 405B), **Mixtral** (8x22B), **Falcon** (40B+) or **BLOOM** (176B) and fine‑tune them for your own tasks &mdash; right from your desktop computer or Google Colab:

```python
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

# Choose any model available at https://health.petals.dev
model_name = "meta-llama/Meta-Llama-3.1-405B-Instruct"

# Connect to a distributed network hosting model layers
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoDistributedModelForCausalLM.from_pretrained(model_name)

# Run the model as if it were on your computer
inputs = tokenizer("A cat sat", return_tensors="pt")["input_ids"]
outputs = model.generate(inputs, max_new_tokens=5)
print(tokenizer.decode(outputs[0]))  # A cat sat on a mat...
```

<p align="center">
    🚀 &nbsp;<b><a href="https://colab.research.google.com/drive/1uCphNY7gfAUkdDrTx21dZZwCOUDCMPw8?usp=sharing">Try now in Colab</a></b>
</p>

🦙 **Want to run Llama?** [Request access](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct) to its weights, then run `huggingface-cli login` in the terminal before loading the model. Or just try it in our [chatbot app](https://chat.petals.dev).

🔏 **Privacy.** Your data will be processed with the help of other people in the public swarm. Learn more about privacy [here](https://github.com/bigscience-workshop/petals/wiki/Security,-privacy,-and-AI-safety). For sensitive data, you can set up a [private swarm](https://github.com/bigscience-workshop/petals/wiki/Launch-your-own-swarm) among people you trust.

💬 **Any questions?** Ping us in [our Discord](https://discord.gg/KdThf2bWVU)!

## Connect your GPU and increase Petals capacity

Petals is a community-run system &mdash; we rely on people sharing their GPUs. You can help serving one of the [available models](https://health.petals.dev) or host a new model from 🤗 [Model Hub](https://huggingface.co/models)!

As an example, here is how to host a part of [Llama 3.1 (405B) Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct) on your GPU:

🦙 **Want to host Llama?** [Request access](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct) to its weights, then run `huggingface-cli login` in the terminal before loading the model.

🐧 **Linux + Anaconda.** Run these commands for NVIDIA GPUs (or follow [this](https://github.com/bigscience-workshop/petals/wiki/Running-on-AMD-GPU) for AMD):

```bash
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install git+https://github.com/bigscience-workshop/petals
python -m petals.cli.run_server meta-llama/Meta-Llama-3.1-405B-Instruct
```

🪟 **Windows + WSL.** Follow [this guide](https://github.com/bigscience-workshop/petals/wiki/Run-Petals-server-on-Windows) on our Wiki.

🐋 **Docker.** Run our [Docker](https://www.docker.com) image for NVIDIA GPUs (or follow [this](https://github.com/bigscience-workshop/petals/wiki/Running-on-AMD-GPU) for AMD):

```bash
sudo docker run -p 31330:31330 --ipc host --gpus all --volume petals-cache:/cache --rm \
    learningathome/petals:main \
    python -m petals.cli.run_server --port 31330 meta-llama/Meta-Llama-3.1-405B-Instruct
```

🍏 **macOS + Apple M1/M2 GPU.** Install [Homebrew](https://brew.sh/), then run these commands:

```bash
brew install python
python3 -m pip install git+https://github.com/bigscience-workshop/petals
python3 -m petals.cli.run_server meta-llama/Meta-Llama-3.1-405B-Instruct
```

<p align="center">
    📚 &nbsp;<b><a href="https://github.com/bigscience-workshop/petals/wiki/FAQ:-Frequently-asked-questions#running-a-server">Learn more</a></b> (how to use multiple GPUs, start the server on boot, etc.)
</p>

🔒 **Security.** Hosting a server does not allow others to run custom code on your computer. Learn more [here](https://github.com/bigscience-workshop/petals/wiki/Security,-privacy,-and-AI-safety).

💬 **Any questions?** Ping us in [our Discord](https://discord.gg/X7DgtxgMhc)!

🏆 **Thank you!** Once you load and host 10+ blocks, we can show your name or link on the [swarm monitor](https://health.petals.dev) as a way to say thanks. You can specify them with `--public_name YOUR_NAME`.

## How does it work?

- You load a small part of the model, then join a [network](https://health.petals.dev) of people serving the other parts. Single‑batch inference runs at up to **6 tokens/sec** for **Llama 2** (70B) and up to **4 tokens/sec** for **Falcon** (180B) — enough for [chatbots](https://chat.petals.dev) and interactive apps.
- You can employ any fine-tuning and sampling methods, execute custom paths through the model, or see its hidden states. You get the comforts of an API with the flexibility of **PyTorch** and **🤗 Transformers**.

<p align="center">
    <img src="https://i.imgur.com/RTYF3yW.png" width="800">
</p>

<p align="center">
    📜 &nbsp;<b><a href="https://arxiv.org/pdf/2209.01188.pdf">Read paper</a></b>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    📚 &nbsp;<b><a href="https://github.com/bigscience-workshop/petals/wiki/FAQ:-Frequently-asked-questions">See FAQ</a></b>
</p>

## 📚 Tutorials, examples, and more

Basic tutorials:

- Getting started: [tutorial](https://colab.research.google.com/drive/1uCphNY7gfAUkdDrTx21dZZwCOUDCMPw8?usp=sharing)
- Prompt-tune Llama-65B for text semantic classification: [tutorial](https://colab.research.google.com/github/bigscience-workshop/petals/blob/main/examples/prompt-tuning-sst2.ipynb)
- Prompt-tune BLOOM to create a personified chatbot: [tutorial](https://colab.research.google.com/github/bigscience-workshop/petals/blob/main/examples/prompt-tuning-personachat.ipynb)

Useful tools:

- [Chatbot web app](https://chat.petals.dev) (connects to Petals via an HTTP/WebSocket endpoint): [source code](https://github.com/petals-infra/chat.petals.dev)
- [Monitor](https://health.petals.dev) for the public swarm: [source code](https://github.com/petals-infra/health.petals.dev)

Advanced guides:

- Launch a private swarm: [guide](https://github.com/bigscience-workshop/petals/wiki/Launch-your-own-swarm)
- Run a custom model: [guide](https://github.com/bigscience-workshop/petals/wiki/Run-a-custom-model-with-Petals)

### Benchmarks

Please see **Section 3.3** of our [paper](https://arxiv.org/pdf/2209.01188.pdf).

### 🛠️ Contributing

Please see our [FAQ](https://github.com/bigscience-workshop/petals/wiki/FAQ:-Frequently-asked-questions#contributing) on contributing.

### 📜 Citations

Alexander Borzunov, Dmitry Baranchuk, Tim Dettmers, Max Ryabinin, Younes Belkada, Artem Chumachenko, Pavel Samygin, and Colin Raffel.
[Petals: Collaborative Inference and Fine-tuning of Large Models.](https://arxiv.org/abs/2209.01188)
_Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)._ 2023.

```bibtex
@inproceedings{borzunov2023petals,
  title = {Petals: Collaborative Inference and Fine-tuning of Large Models},
  author = {Borzunov, Alexander and Baranchuk, Dmitry and Dettmers, Tim and Riabinin, Maksim and Belkada, Younes and Chumachenko, Artem and Samygin, Pavel and Raffel, Colin},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)},
  pages = {558--568},
  year = {2023},
  url = {https://arxiv.org/abs/2209.01188}
}
```

Alexander Borzunov, Max Ryabinin, Artem Chumachenko, Dmitry Baranchuk, Tim Dettmers, Younes Belkada, Pavel Samygin, and Colin Raffel.
[Distributed inference and fine-tuning of large language models over the Internet.](https://arxiv.org/abs/2312.08361)
_Advances in Neural Information Processing Systems_ 36 (2023).

```bibtex
@inproceedings{borzunov2023distributed,
  title = {Distributed inference and fine-tuning of large language models over the {I}nternet},
  author = {Borzunov, Alexander and Ryabinin, Max and Chumachenko, Artem and Baranchuk, Dmitry and Dettmers, Tim and Belkada, Younes and Samygin, Pavel and Raffel, Colin},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {36},
  pages = {12312--12331},
  year = {2023},
  url = {https://arxiv.org/abs/2312.08361}
}
```

--------------------------------------------------------------------------------

<p align="center">
    This project is a part of the <a href="https://bigscience.huggingface.co/">BigScience</a> research workshop.
</p>
<p align="center">
    <img src="https://petals.dev/bigscience.png" width="150">
</p>
