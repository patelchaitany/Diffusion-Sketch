#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

# ── Colors ───────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }

# ── 1. Virtual environment ──────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment in $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
else
    info "Virtual environment already exists."
fi

source "$VENV_DIR/bin/activate"
info "Using Python: $(python --version) at $(which python)"

# ── 2. Upgrade pip ──────────────────────────────────────────────────
info "Upgrading pip ..."
pip install --upgrade pip --quiet

# ── 3. Install package (editable) + dev dependencies ────────────────
info "Installing diffusion-sketch in editable mode ..."
pip install -e ".[dev]" --quiet

# ── 4. Create data directories ──────────────────────────────────────
info "Creating data directories ..."
mkdir -p data/train data/val

# ── 5. Create output directories ────────────────────────────────────
mkdir -p checkpoints samples ray_results

# ── 6. Verify installation ──────────────────────────────────────────
info "Verifying installation ..."
python -c "import diffusion_sketch; print(f'diffusion-sketch v{diffusion_sketch.__version__} installed.')"

# ── 7. Run tests ────────────────────────────────────────────────────
info "Running tests ..."
if pytest tests/ -q; then
    info "All tests passed."
else
    warn "Some tests failed — check output above."
fi

echo ""
info "Setup complete. Activate the environment with:"
echo "    source $VENV_DIR/bin/activate"
echo ""
info "Place your paired images in data/train/ and data/val/, then run:"
echo "    python -m diffusion_sketch"
echo ""
info "Override any config value inline:"
echo "    python -m diffusion_sketch training.epochs=50 training.batch_size=8"
