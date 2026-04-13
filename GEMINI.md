# gReLU Environment Setup

This project uses a dedicated conda environment named `grelu_dev` located at `~/.conda/envs/grelu_dev`. 

## Build & Dependencies
- The environment requires **Python 3.12** to be compatible with the `alphagenome-pytorch` submodule.
- Both the main `gReLU` package and the `alphagenome-pytorch` submodule are installed in editable mode.
- Testing dependencies (e.g., `pytest`, `pytest-cov`) have been installed via `mamba`.

To rebuild or update the environment, use the following commands:
```bash
# 1. Ensure Python 3.12 via mamba
~/.conda/envs/mamba_env/bin/mamba install -n grelu_dev python=3.12 -y

# 2. Activate
source activate.sh

# 3. Install the submodule in editable mode
pip install -e ./src/alphagenome_pytorch

# 4. Install the main project in editable mode
pip install -e .
```

## Activation
To activate the environment, source the included script from the project root:
```bash
source activate.sh
```
Or use conda directly:
```bash
conda activate grelu_dev
```

## Documentation & Notes
- `tutorial_list.md`: An overview of the Jupyter Notebook tutorials available in the project.
- `arch.md`: Documentation on the gReLU model architecture and how to integrate external PyTorch models.
- `alphagenome.md`: Notes and hardware optimization strategies for fine-tuning the AlphaGenome model.

## Project Memory
- **`MEMORY.md`**: A chronological changelog of major architectural decisions and feature integrations. **MANDATORY**: Whenever you complete a significant codebase change, integration, or structural refactor, you MUST append an entry to `MEMORY.md` documenting the objective, key implementations, and validation steps.
