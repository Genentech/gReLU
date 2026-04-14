# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

---

## Lab Environment Constraints

This environment is part of a shared laboratory cluster. The following rules are **mandatory**.

- **No `sudo`**: Privilege escalation is strictly forbidden. All operations must stay within `~` (`/home/$USER`).
- **Scope**: Do not modify system-wide files (`/etc`, `/usr/lib`, `/var`).
- **Toolchain**: `mise` manages runtimes (Node.js, Python, Go, etc.). Tools live in `~/.local/bin/` and `~/.local/share/mise/`. Prefer portable binaries; avoid compiling from source unless unavoidable.
- **Shell**: Primary shell is a locally installed Zsh (`~/.local/bin/zsh`). Do NOT break the Bash→Zsh handoff logic in `~/.bashrc`.
- **GPUs**: Monitor with `nvidia-smi` or `show_load`. Use judiciously.

### Conda/Mamba Rules

- **Never** install packages or run programs in the `base` Conda environment.
- Always create isolated environments: `conda create -n <env> python=3.x`.
- For fast solving, use the dedicated solver env: `~/.conda/envs/mamba_env/bin/mamba`.
- In non-interactive scripts, source conda first: `source /work/miniconda3/etc/profile.d/conda.sh` before `conda activate`.

---

## gReLU Project Environment

The project uses a dedicated conda environment **`grelu_dev`** at `~/.conda/envs/grelu_dev`.

### Activate

```bash
source activate.sh          # from project root
# or
conda activate grelu_dev
```

### Rebuild / Update

```bash
# 1. Ensure Python 3.12
~/.conda/envs/mamba_env/bin/mamba install -n grelu_dev python=3.12 -y

# 2. Activate
source activate.sh

# 3. Install AlphaGenome submodule (editable)
pip install -e ./src/alphagenome_pytorch

# 4. Install gReLU (editable)
pip install -e .
```

### Key Docs

| File | Purpose |
|---|---|
| `tutorial_list.md` | Overview of Jupyter Notebook tutorials |
| `arch.md` | gReLU model architecture & external PyTorch model integration |
| `alphagenome.md` | AlphaGenome fine-tuning notes and GPU optimization |
| `MEMORY.md` | Chronological changelog of major architectural decisions |

**MANDATORY**: After any significant codebase change, integration, or structural refactor, append an entry to `MEMORY.md` documenting the objective, key implementations, and validation steps.

---

## AlphaGenome-PyTorch Submodule

Located at `src/alphagenome_pytorch/`. See its own `CLAUDE.md` for detailed architecture, commands, and test strategy.

Quick reference:

```bash
# Run unit tests (no JAX required)
pytest tests/unit/ -v

# Run PyTorch-only integration tests
pytest tests/integration/ -v --torch-weights=model.pth

# Convert JAX weights to PyTorch
python scripts/convert_weights.py --input jax_checkpoint --output model.pth
```
