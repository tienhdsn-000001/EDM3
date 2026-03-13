#!/usr/bin/env bash
# ==========================================================================
# EEPM3: Production Sparse Training Pipeline
#
# Runs the full Phase 5 SOTA pipeline:
#   1. Generate trajectories (Conv1D dual-head policy, T=2.0)
#   2. Score via AlphaGenome API (DNASE, 131072-bp padded, async)
#   3. RBS data augmentation (top-10% backward synthesis)
#   4. Offline α-GFN training (Sub-EB + value head)
#
# Works on: Kaggle, Colab, and local machines.
#
# Usage:
#   # Kaggle (set API key in Secrets):
#   bash run_overnight.sh
#
#   # Local:
#   export ALPHA_GENOME_API_KEY="your-key"
#   bash run_overnight.sh
# ==========================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ── Platform Detection ────────────────────────────────────────
if [ -d "/content" ]; then
    PLATFORM="Colab"
    # Fallback logic: Use Drive if mounted, otherwise use local /content/data
    if [ -d "/content/drive/MyDrive" ]; then
        DATA_DIR="/content/drive/MyDrive/EEPM3_Data"
        echo "[COLAB] Persistence: Google Drive enabled."
    else
        DATA_DIR="/content/data"
        echo "[COLAB] Warning: Drive not mounted. Data will be lost if session disconnects."
    fi
    # --- Hardened Persistence: Symlink core folders to Drive ---
    for dir in "data" "logs" "checkpoints"; do
        mkdir -p "${DATA_DIR}/${dir}"
        if [ -d "$dir" ] && [ ! -L "$dir" ]; then
            # If a local dir exists but isn't a symlink, merge it to Drive then remove
            cp -rn "$dir"/. "${DATA_DIR}/${dir}/" 2>/dev/null || true
            rm -rf "$dir"
        fi
        ln -sfn "${DATA_DIR}/${dir}" "$dir"
        echo "[COLAB] Persistence: $dir -> ${DATA_DIR}/${dir}"
    done
    # Pull API key from Colab Secrets (if available)
    SECRET=$(python -c "try: from google.colab import userdata; print(userdata.get('ALPHA_GENOME_API_KEY')); except: pass" 2>/dev/null || echo "")
    if [ -n "$SECRET" ]; then
        ALPHA_GENOME_API_KEY="$SECRET"
    fi
elif [ -d "/kaggle" ]; then
    PLATFORM="Kaggle"
    DATA_DIR="/kaggle/working/data"
    # Kaggle sets secrets as env vars if correctly mapped
else
    PLATFORM="Local"
    DATA_DIR="${PROJECT_DIR}/data"
fi
echo "[Platform] ${PLATFORM}"

# ── Environment Setup ─────────────────────────────────────────
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "[ENV] Virtual environment activated."
fi

# Install dependencies if missing
echo "[SETUP] Checking system dependencies..."
python -c "import jax, flax, optax" 2>/dev/null || {
    echo "  -> [INSTALL] JAX/Flax optimization suite..."
    pip install jax flax optax 2>&1 | tail -n 5
}
python -c "import alphagenome" 2>/dev/null || {
    echo "  -> [INSTALL] AlphaGenome SDK..."
    pip install alphagenome 2>&1 | tail -n 5
}
python -c "import torch, evo2" 2>/dev/null || {
    echo "  -> [INSTALL] PyTorch and Evo2 (Large Install)..."
    echo "     Note: This can take several minutes due to model framework size."
    pip install torch flash-attn evo2 2>&1 | tail -n 10
}
echo "[SETUP] Dependency check complete."

# ── API Key Validation ────────────────────────────────────────
if [ -z "${ALPHA_GENOME_API_KEY:-}" ]; then
    # Try Kaggle secrets
    ALPHA_GENOME_API_KEY=$(python -c "
try:
    from kaggle_secrets import UserSecretsClient
    print(UserSecretsClient().get_secret('ALPHA_GENOME_API_KEY'))
except: pass
" 2>/dev/null || true)
fi

if [ -z "${ALPHA_GENOME_API_KEY:-}" ]; then
    echo "[ERROR] ALPHA_GENOME_API_KEY not set."
    echo "  Kaggle: Add to Secrets (Add-ons → Secrets)"
    echo "  Local:  export ALPHA_GENOME_API_KEY=your_key"
    exit 1
fi
export ALPHA_GENOME_API_KEY
echo "[AUTH] API key loaded (${ALPHA_GENOME_API_KEY:0:8}...)"

# ── Directories ───────────────────────────────────────────────
mkdir -p "${DATA_DIR}" checkpoints logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/pipeline_${TIMESTAMP}.log"

echo ""
echo "=========================================="
echo "EEPM3 Phase 5: SOTA Sparse Training Pipeline"
echo "=========================================="
echo "Start: $(date)"
echo "Log:   ${LOG_FILE}"
echo ""

exec > >(tee -a "$LOG_FILE") 2>&1

# ── Step 1: Trajectory Generation ─────────────────────────────
echo "=========================================="
echo "STEP 1: Trajectory Generation"
echo "=========================================="
if [ -f "data/unscored_trajectories.npz" ]; then
    echo "[SKIP] Found existing trajectories in persistent data storage."
else
    # Check if user provided them manually in the root Drive folder or local
    if [ -f "${DATA_DIR}/unscored_trajectories.npz" ]; then
         cp "${DATA_DIR}/unscored_trajectories.npz" "data/unscored_trajectories.npz"
         echo "[COPY] Imported trajectories from ${DATA_DIR}"
    else
        NUM_TRAJ=${EEPM3_NUM_TRAJECTORIES:-5000}
        echo "Generating ${NUM_TRAJ} trajectories (T=2.0, dual-head Conv1D)..."
        python src/1_trajectory_sampler.py "$NUM_TRAJ" || {
            echo "[FATAL] Trajectory generation failed."
            exit 1
        }
    fi
fi
echo "[✓] Step 1 complete."

# ── Step 2: AlphaGenome API Scoring ───────────────────────────
echo ""
echo "=========================================="
echo "STEP 2: AlphaGenome API Scoring (DNASE)"
echo "=========================================="
echo "Scoring sequences (131,072-bp padded, async)..."
echo "Crash-safe: resumes from SQLite on restart."

python src/2_api_worker.py || {
    echo "[WARN] API worker exited with errors."
    echo "  Trainer will use whatever was scored."
}
echo "[✓] Step 2 complete."

# ── Step 3: RBS Augmentation ─────────────────────────────────
echo ""
echo "=========================================="
echo "STEP 3: RBS Data Augmentation"
echo "=========================================="
echo "Hallucinating trajectories for top-10% rewards..."

python src/4_rbs_augmenter.py || {
    echo "[WARN] RBS augmentation failed. Training on original data."
}
echo "[✓] Step 3 complete."

# ── Step 4: Offline α-GFN Training ───────────────────────────
echo ""
echo "=========================================="
echo "STEP 4: Offline α-GFN Training (Sub-EB)"
echo "=========================================="
echo "Training with α=${ALPHA_GFN:-0.5}, dual-head policy..."

# Use V2 trainer if available, fallback to V1
if [ -f "offline_trainer_v2.py" ]; then
    python src/offline_trainer_v2.py
else
    python src/3_offline_trainer.py
fi
echo "[✓] Step 4 complete."

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
echo "End: $(date)"
echo ""
echo "Artifacts:"
echo "  Trajectories:     data/unscored_trajectories.npz"
echo "  Replay Buffer:    data/experience_replay.db"
echo "  Augmented Buffer: data/experience_replay_augmented.db"
echo "  Checkpoints:      checkpoints/"
echo "  Log:              ${LOG_FILE}"
echo ""
echo "Inspect replay buffer:"
echo "  python -c \"import sqlite3; c=sqlite3.connect('data/experience_replay.db'); print(c.execute('SELECT COUNT(*), AVG(reward), MIN(reward), MAX(reward) FROM experiences').fetchone())\""
echo "=========================================="
