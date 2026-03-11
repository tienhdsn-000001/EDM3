# EDM3 — Expandable Demographic Mimicry by Mutation Module for AlphaGenome

> **GFlowNet-driven DNA sequence editing for demographic-specific epigenetic profile mimicry, powered by the real AlphaGenome API.**

---

## Overview

EDM3 uses **Generative Flow Networks (GFlowNets)** to learn a mutation policy for editing 100kb DNA sequences. A foundation model (AlphaGenome) evaluates the epigenetic consequences of each edit, and the GFlowNet learns to propose mutations whose predicted profiles match a target demographic (age, tissue, ancestry).

### Core Architecture (Phase 5 — SOTA)

```
Wild-type DNA → [GFlowNet MDP: 10 mutations] → Mutated DNA
                                                     ↓
                                        AlphaGenome API (DNASE)
                                                     ↓
                                      R(x) = exp(-α·L_mask) + β·P_Evo
                                                     ↓
                               α-GFN Trajectory Balance + Sub-EB Loss
```

**Key innovations:**
- **Dual-head Conv1D policy** (34,136 params): Factored action head + value head V(s)
- **α-GFN loss**: Tunable forward/backward mixing (α=0.5)
- **Sub-EB objective**: Sub-trajectory evaluation balance with V(s) bridging
- **RBS augmentation**: Retrospective Backward Synthesis on top-10% trajectories
- **Partial GFlowNet**: 10kb windowed action space (12.5× reduction)

---

## 🚀 Quick Start — Run on Kaggle (Recommended)

The pipeline is decoupled into three notebooks to protect GPU quota from the slow AlphaGenome API.

### Phase 6: Decoupled "Fire & Forget" Execution

1. **Step 1: Generation (`notebooks/nb_A_generator.ipynb`)**
   - **Hardware**: GPU T4 ×2
   - **Action**: Run all. Generates 5,000 trajectories using the dual-head Conv1D policy.
   - **Output**: Export `/kaggle/working` as a dataset named `unscored-trajectories`.

2. **Step 2: API Worker (`notebooks/nb_B_api_worker.ipynb`)**
   - **Hardware**: **CPU ONLY** (Accelerator: None)
   - **Data**: Mount the `unscored-trajectories` dataset to `../input/unscored-trajectories/`.
   - **Action**: Run via "Save Version -> Save & Run All (Commit)".
   - **Resilience**: Runs asynchronously in the background for up to 12 hours. Uses SQLite WAL to save incrementally.
   - **Output**: Export `/kaggle/working` as a dataset named `edm3-experience-buffer`.

3. **Step 3: Offline Training (`notebooks/nb_C_offline_trainer.ipynb`)**
   - **Hardware**: GPU T4 ×2
   - **Data**: Mount the `edm3-experience-buffer` database.
   - **Action**: Run all. Executes RBS augmentation and SOTA α-GFN offline training.

### Alternative: Bash Pipeline on Kaggle

Upload the entire repo as a Kaggle dataset, then in a notebook cell:

```bash
!export ALPHAGENOME_API_KEY="your-key" && bash run_overnight.sh
```

### Alternative: Google Colab

Open `notebooks/edm3_kaggle_pipeline.ipynb` in Colab. Set the API key via:
```python
from google.colab import userdata
# Add ALPHAGENOME_API_KEY in Colab Secrets (key icon in left panel)
```

---

## Pipeline Architecture

### Decoupled 4-Stage Pipeline

| Stage | Script | Purpose | Runtime |
|-------|--------|---------|---------|
| 1 — Generate | `1_trajectory_sampler.py` | Sample trajectories from Conv1D policy (T=2.0) | ~8 min (5k traj) |
| 2 — Score | `2_api_worker.py` | Async AlphaGenome API DNASE scoring | ~30 min (5k traj) |
| 3 — Augment | `4_rbs_augmenter.py` | RBS: hallucinate trajectories for top-10% | <1 min |
| 4 — Train | `offline_trainer_v2.py` | α-GFN + Sub-EB offline training | ~5 min (200 epochs) |

### Phase 5 SOTA Upgrades

| Feature | File | Description |
|---------|------|-------------|
| Dual-Head Policy | `offline_trainer_v2.py` | Action head + V(s) value head (34,136 params) |
| α-GFN Loss | `offline_trainer_v2.py` | Mixing parameter α for exploration/exploitation |
| Sub-EB | `offline_trainer_v2.py` | Sub-trajectory evaluation with value bridging |
| RBS Augmentation | `4_rbs_augmenter.py` | Mutation permutation on high-reward sequences |
| Partial GFlowNet | `trajectory_sampler_v2.py` | 10kb windowed action space (12.5× reduction) |

---

## Modules

### Core

| File | Description |
|------|-------------|
| `gflownet_env.py` | MDP environment + Conv1D GeneratorPolicy (20,311 params) |
| `gflownet_trainer.py` | TB loss, reward function, deterministic proxy oracles |
| `training_loop.py` | VMAP-batched training with gradient clipping + EMA convergence |
| `data_pipeline.py` | GTEx BigWig → (B, 7812, 5930) tensor pipeline |

### Production Pipeline

| File | Description |
|------|-------------|
| `1_trajectory_sampler.py` | Offline trajectory generation with temperature scaling |
| `2_api_worker.py` | Async AlphaGenome API scorer (DNASE, 131072-bp, SQLite buffer) |
| `3_offline_trainer.py` | Offline TB trainer from replay buffer |
| `4_rbs_augmenter.py` | Retrospective Backward Synthesis data augmenter |
| `offline_trainer_v2.py` | α-GFN + Sub-EB SOTA trainer with dual-head policy |
| `trajectory_sampler_v2.py` | Partial GFlowNet with 10kb windowed action masking |
| `run_overnight.sh` | Orchestration script (Kaggle/Colab/Local auto-detection) |

### Notebooks (Phase 6 Decoupled)

| File | Description |
|------|-------------|
| `nb_A_generator.ipynb` | Fast GPU Trajectory Generation (5000 seq_len=100k) |
| `nb_B_api_worker.ipynb` | Slow CPU Async AlphaGenome API Worker (WAL-sqlite) |
| `nb_C_offline_trainer.ipynb` | Fast GPU RBS Augmentation and α-GFN Offline Training |

---

## AlphaGenome API Integration

**API method:** `predict_sequence(sequence, requested_outputs=[OutputType.DNASE])`

**Key constraints discovered during integration:**
- Sequences must be exactly one of: `[16384, 131072, 524288, 1048576]` bp
- Our 100,000 bp sequences are N-padded to **131,072 bp**
- Different modalities (DNASE, CHIP_HISTONE, ATAC) return at different bin resolutions — only DNASE is used
- Async scoring via `run_in_executor` with exponential backoff for RESOURCE_EXHAUSTED

**Reward function:**
```
R(x) = exp(-α · MSE(DNASE_pred)) + β · mean(|DNASE_pred|)
```

---

## Tests

```bash
python test_env.py       # MDP environment, Conv1D policy, parameter count
python test_trainer.py   # Deterministic oracles, reward, gradient flow
python test_pipeline.py  # BigWig parsing, tensor shapes, mask integrity
```

---

## Requirements

- Python 3.10+
- JAX (with GPU/TPU for training)
- Flax, Optax
- `alphagenome` (pip install alphagenome)
- NumPy, Pandas, aiohttp, sqlite3

```bash
pip install jax[cuda12] flax optax alphagenome numpy pandas aiohttp pyBigWig
```

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Sequence Length | 100,000 bp (padded to 131,072 for API) |
| Policy Parameters (V1) | 20,311 |
| Policy Parameters (V2 dual-head) | 34,136 |
| Parameter Budget | < 5,000,000 ✅ |
| Edits per Trajectory | 10 |
| Window Size (Partial GFN) | 10,000 bp (12.5× action space reduction) |
| Gradient Clipping | Global norm 1.0 |
| Optimizer | AdamW (lr=1e-4) |
| Convergence Detection | EMA (α=0.95, 5% drop threshold) |

---

## Project Documents

| Document | Description |
|----------|-------------|
| [`EXECUTIVE_REPORT.md`](EXECUTIVE_REPORT.md) | Independent audit with red/blue team analysis |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | Module completion tracking and phase roadmap |
| [`EXPERIMENTAL_RIGOR_AUDIT.md`](EXPERIMENTAL_RIGOR_AUDIT.md) | Scientific methodology assessment |

---

## License

Research prototype — not for clinical or production use.
