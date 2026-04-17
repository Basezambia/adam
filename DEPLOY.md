# ADAM — Deployment Guide

Three places to push ADAM. **Local demo already works on http://localhost:8000/**

---

## 1 · GitHub (code repo)

**Create the repo** at https://github.com/new (name `adam`, empty, no README).

Then, locally:
```bash
cd C:/Users/lisel/hexq
git remote add origin https://github.com/YOUR_USERNAME/adam.git
git push -u origin main
```

Done. The full codebase (18 files, ~194KB) is live.

---

## 2 · HuggingFace Hub (the 204MB checkpoint)

The trained weights are too big for GitHub. Upload to HF Hub:

```bash
pip install huggingface_hub
huggingface-cli login                 # paste your HF token
huggingface-cli upload YOUR_HF_USER/adam adam_checkpoint.pt
```

People can then download it with:
```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("YOUR_HF_USER/adam", "adam_checkpoint.pt")
```

---

## 3 · HuggingFace Spaces (the public live demo)

Turns the local demo into a public URL anyone can play with.

```bash
cd C:/Users/lisel/hexq/hf_space

# copy the files needed by the Dockerfile into this folder
cp ../adam.py ../demo_server.py .
cp -r ../demo .

# push to HF Spaces
huggingface-cli login
huggingface-cli repo create adam-demo --type space --space_sdk docker
git init -b main
git remote add origin https://huggingface.co/spaces/YOUR_HF_USER/adam-demo
git add .
git commit -m "Initial ADAM demo Space"
git push -u origin main
```

Then set a Space secret so the container can fetch your checkpoint:
- Go to `https://huggingface.co/spaces/YOUR_HF_USER/adam-demo/settings`
- **Variables and secrets** → add: `ADAM_HF_REPO = YOUR_HF_USER/adam`

First boot takes ~90s (downloads torch + checkpoint). After that, anyone
visiting the Space URL gets the full 4-tab interactive demo.

---

## 4 · Vercel (for static sites you already have)

Already deployed:
- Paper: `vercel deploy --prod --yes` from `paper_deploy/`
- Chat UI: `vercel deploy --prod --yes` from `adam_chat/`
