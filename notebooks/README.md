# Notebooks

Exercises for each part of the curriculum. Each lesson's "Try it" box names a file here, for example `notebooks/01-foundations/gradient-descent.ipynb`.

Suggested layout:

```text
notebooks/
├── 01-foundations/
├── 02-classical-ml/
├── 03-neural-networks/
├── 04-architectures/
├── 05-training/
├── 06-open-models/
└── 07-rl/
```

## Environment

See the site's "Your hardware" section for setup on Windows + WSL2, Ubuntu or macOS. Then, from the repo root:

```bash
uv venv --python 3.12
source .venv/bin/activate
# Windows (WSL2) or Ubuntu with the RTX 3070
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install -r notebooks/requirements.txt

# macOS on Apple Silicon (bitsandbytes is CUDA-only, so skip it and add MLX)
uv pip install torch torchvision mlx mlx-lm
grep -v bitsandbytes notebooks/requirements.txt > /tmp/requirements-mac.txt
uv pip install -r /tmp/requirements-mac.txt
```

Pick the device at the top of each notebook so the same code runs on both machines:

```python
import torch
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
```
