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
└── 06-open-models/
```

## Environment

See the site's "Setting up your machine" page for the GPU setup. Then, from the repo root:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install -r notebooks/requirements.txt
```
