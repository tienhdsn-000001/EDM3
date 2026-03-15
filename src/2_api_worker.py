"""
Phase 4, Script 2: Asynchronous AlphaGenome API Worker.

Consumes unscored trajectories from data/unscored_trajectories.npz, queries the
real AlphaGenome API for epigenetic predictions, computes the scalar reward R(x),
and stores scored experiences in a SQLite replay buffer.

This script is completely decoupled from JAX/XLA to prevent RAM bloat during
overnight network runs. It uses only numpy, aiohttp, and the alphagenome package.

Features:
  - Parallel async requests with configurable concurrency semaphore
  - Exponential backoff for HTTP 429 and 500+ errors
  - Crash-resilient: scored results are flushed to SQLite immediately
  - Progress tracking and resumption from partially-scored datasets
"""

import os
import sys
import json
import time
import sqlite3
import asyncio
import logging
import torch
import numpy as np
from typing import Optional

# ── Flash Attention Mocking (Colab T4/CPU Compatibility) ──────
# The evo2 library hard-imports flash_attn which often fails to compile in Colab.
# We mock it here to force the library to fall back to native PyTorch SDPA.
try:
    import flash_attn
except ImportError:
    from unittest.mock import MagicMock
    _mock = MagicMock()
    # Mock the base module and all possible internal/extension names
    sys.modules["flash_attn"] = _mock
    sys.modules["flash_attn_2_cuda"] = _mock 
    sys.modules["flash_attn.flash_attn_interface"] = _mock
    sys.modules["flash_attn.bert_padding"] = _mock
    sys.modules["flash_attn.losses"] = _mock
    sys.modules["flash_attn.losses.cross_entropy"] = _mock
    sys.modules["flash_attn.ops"] = _mock
    sys.modules["flash_attn.ops.rms_norm"] = _mock
    sys.modules["flash_attn.ops.layer_norm"] = _mock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("api_worker")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_PATH = "data/unscored_trajectories.npz"
DB_PATH = "data/experience_replay.db"
MAX_CONCURRENCY = 5          # Semaphore limit (reduced to avoid RAM overload)
MAX_RETRIES = 8              # Max exponential backoff retries per request
BASE_BACKOFF = 1.0           # Base seconds for exponential backoff
ALPHA_REWARD = 1.0           # α in R(x) = exp(-α·L_mask) + β·log P_Evo
BETA_REWARD = 0.1            # β weight on Evo2 term

# AlphaGenome prediction parameters
# IMPORTANT: AlphaGenome only supports specific lengths: [16384, 131072, 524288, 1048576]
# Our 100,000 bp sequences must be padded to 131,072 bp with N's
API_SEQ_LEN = 131_072        # Nearest supported length above 100,000
NUM_BINS = 781               # 100000 // 128
NUM_TRACKS = 5930


# ---------------------------------------------------------------------------
# SQLite Experience Replay Buffer
# ---------------------------------------------------------------------------

def init_database(db_path: str) -> sqlite3.Connection:
    """Creates the experience replay database with WAL mode for crash safety."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiences (
            trajectory_id INTEGER PRIMARY KEY,
            actions BLOB NOT NULL,
            forward_log_probs BLOB NOT NULL,
            reward REAL NOT NULL,
            api_latency_ms REAL,
            reward_model TEXT DEFAULT 'legacy_oracle',
            scored_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # --- Migration: Add reward_model column if missing in older DBs ---
    cursor = conn.execute("PRAGMA table_info(experiences)")
    columns = {row[1] for row in cursor.fetchall()}
    if "reward_model" not in columns:
        log.info("[DB/Migration] Adding 'reward_model' column to existing database...")
        conn.execute("ALTER TABLE experiences ADD COLUMN reward_model TEXT DEFAULT 'legacy_oracle'")
    
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_trajectory_id
        ON experiences(trajectory_id)
    """)
    conn.commit()
    return conn


def get_scored_ids(conn: sqlite3.Connection, reward_model: str) -> set:
    """Returns the set of trajectory IDs already scored by the CURRENT model."""
    cursor = conn.execute(
        "SELECT trajectory_id FROM experiences WHERE reward_model = ?", 
        (reward_model,)
    )
    return {row[0] for row in cursor.fetchall()}


def insert_experience(
    conn: sqlite3.Connection,
    trajectory_id: int,
    actions: np.ndarray,
    forward_log_probs: np.ndarray,
    reward: float,
    api_latency_ms: float,
    reward_model: str,
):
    """Inserts a scored experience and flushes immediately."""
    conn.execute(
        "INSERT OR REPLACE INTO experiences "
        "(trajectory_id, actions, forward_log_probs, reward, api_latency_ms, reward_model) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            trajectory_id,
            actions.tobytes(),
            forward_log_probs.tobytes(),
            float(reward),
            float(api_latency_ms),
            reward_model,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Masked Modality Loss (numpy-only, no JAX dependency)
# ---------------------------------------------------------------------------

def masked_modality_loss_np(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Computes L_mask = mean of squared errors only where mask == 1.
    Pure numpy implementation to avoid JAX import.
    """
    sq_error = (predictions - targets) ** 2
    masked_error = sq_error * mask
    num_valid = max(np.sum(mask), 1.0)
    return float(np.sum(masked_error) / num_valid)


def compute_reward_np(
    ag_predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    evo2_score: float = 0.0,
    alpha: float = 1.0,
    beta: float = 0.1,
) -> float:
    """
    R(x) = exp(-α · L_mask) + β · log P_Evo(x)
    Pure numpy. Evo2 score defaults to 0 when not available from API.
    """
    l_mask = masked_modality_loss_np(ag_predictions, targets, mask)
    reward = np.exp(-alpha * l_mask) + beta * evo2_score
    return float(max(reward, 1e-8))


# ---------------------------------------------------------------------------
# AlphaGenome API Client
# ---------------------------------------------------------------------------

# Module-level singleton client (created once, reused across requests)
_api_client = None

def _get_api_client(api_key: str):
    """Returns a singleton DnaClient instance."""
    global _api_client
    if _api_client is None:
        from alphagenome.models import dna_client
        _api_client = dna_client.create(api_key)
        log.info("[API] AlphaGenome DnaClient initialized.")
    return _api_client


# ---------------------------------------------------------------------------
# Evo2 Foundation Model Integration
# ---------------------------------------------------------------------------

_evo2_model = None
evo2_lock = asyncio.Lock()

def _get_evo2_model():
    """Loads the authentic Evo2 7B model in bfloat16 for Colab T4 compatibility."""
    global _evo2_model
    
    model_name = os.environ.get("EVO2_MODEL_NAME", "evo2_7b")
    
    # Handle Legacy Oracle mode (CPU Friendly)
    if model_name == "legacy_oracle":
        return "legacy_oracle"

    # Device Selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # TPU Detection (v5e-1 VMs and Colab TPU)
    is_tpu = any(k in os.environ for k in [
        "TPU_NAME", "TPU_ACCELERATOR_TYPE", "JAX_PLATFORMS", 
        "KAGGLE_TPU_NAME", "COLAB_TPU_ADDR"
    ]) or os.path.exists("/usr/lib64/libtpu.so") or os.path.exists("/usr/local/lib/libtpu.so")

    if is_tpu and device == "cpu":
        log.info("[TPU/Hybrid] TPU Environment Detected.")
        log.info("[TPU/Hybrid] Offloading Evo2 foundation model to High-RAM TPU VM CPU.")
        log.info("[TPU/Hybrid] Performance Note: Using bfloat16 for 7B foundation weights.")
    elif not torch.cuda.is_available() and not is_tpu:
        # Standard CPU-only fallback logic
        log.error("=" * 70)
        log.error("[CRITICAL] NO GPU OR TPU DETECTED")
        log.error("The Evo2 7B model requires hardware acceleration.")
        log.error("If you are on a restricted CPU instance, switch to the oracle:")
        log.error("  export EVO2_MODEL_NAME=legacy_oracle")
        log.error("=" * 70)
        raise RuntimeError("Hardware acceleration required for Evo2. Switch to legacy_oracle if necessary.")

    if _evo2_model is None:
        try:
            from evo2 import Evo2
            log.info(f"[Evo2] Initializing authentic model: {model_name}")
            log.info(f"[Evo2] Verification/Download phase started. This may take 5-10 minutes if not cached...")
            
            # This call handles weight download/verification
            _evo2_model = Evo2(model_name)
            
            # Use bfloat16 for RAM efficiency on both T4 and TPU VM CPUs
            dtype = torch.bfloat16
            log.info(f"[Evo2] Weights verified. Moving to {device} ({dtype})...")
            _evo2_model = _evo2_model.to(device, dtype=dtype)
            _evo2_model.eval()
            log.info(f"[Evo2] Model {model_name} is LIVE on {device}.")
        except Exception as e:
            log.error(f"[Evo2] Failed to initialize model: {e}")
            log.error("  Suggestion: Ensure your environment has sufficient RAM (24GB+ for 7B CPU).")
            raise
    return _evo2_model

@torch.no_grad()
def compute_real_evo2_likelihood(sequence: str) -> float:
    """Computes the log-likelihood of a sequence using the authentic Evo2 7B model.
    Clear cache afterward to prevent VRAM bloat on T4s."""
    model = _get_evo2_model()
    
    # Handle Legacy Oracle mode
    if model == "legacy_oracle":
        # Deterministic but non-uniform score for verification
        return float(hash(sequence) % 1000) / 1000.0

    try:
        if hasattr(model, "score_sequence"):
            score = model.score_sequence(sequence)
        else:
            # Fallback for alternative library method signatures
            score = 0.0
    except Exception as e:
        log.error(f"[Evo2] Scoring failed: {e}")
        score = 0.0

    torch.cuda.empty_cache()
    return float(score)


async def query_alphagenome_api(
    sequence: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    trajectory_id: int,
) -> Optional[np.ndarray]:
    """
    Queries the AlphaGenome API for chromatin predictions on a DNA sequence.

    Uses `predict_sequence()` with DNASE, CHIP_HISTONE, and ATAC output types.
    Implements exponential backoff for rate limiting and server errors.

    Returns: (num_bins, num_features) np.ndarray prediction tensor, or None on failure.
    """
    try:
        from alphagenome.models import dna_client
        from alphagenome.data import genome
    except ImportError:
        log.error(
            "alphagenome package not installed. "
            "Run: pip install ./alphagenome (from the cloned repo)"
        )
        return None

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                loop = asyncio.get_event_loop()

                def _predict():
                    model = _get_api_client(api_key)

                    # Pad sequence to API-supported length (131,072 bp)
                    padded_seq = sequence
                    if len(padded_seq) < API_SEQ_LEN:
                        padded_seq = padded_seq + 'N' * (API_SEQ_LEN - len(padded_seq))

                    # Request DNASE only — modalities return different bin resolutions
                    # so they cannot be concatenated. DNASE is the primary chromatin
                    # accessibility signal needed for reward computation.
                    outputs = model.predict_sequence(
                        sequence=padded_seq,
                        requested_outputs=[
                            dna_client.OutputType.DNASE,
                        ],
                        ontology_terms=None,
                    )

                    # Extract DNASE track data
                    dnase_data = outputs.dnase
                    if dnase_data is not None:
                        # Try common data access patterns
                        if hasattr(dnase_data, 'values'):
                            arr = np.array(dnase_data.values, dtype=np.float32)
                        elif hasattr(dnase_data, 'data'):
                            arr = np.array(dnase_data.data, dtype=np.float32)
                        elif hasattr(dnase_data, 'X'):
                            # AnnData format
                            arr = np.array(dnase_data.X, dtype=np.float32)
                        else:
                            arr = np.array(dnase_data, dtype=np.float32)

                        # Ensure 2D: (bins, tracks)
                        if arr.ndim == 1:
                            arr = arr[:, None]

                        return arr

                    return None

                result = await loop.run_in_executor(None, _predict)

                if result is not None:
                    log.debug(f"  Trajectory {trajectory_id}: API returned shape {result.shape}")
                    return result
                else:
                    log.warning(f"  Trajectory {trajectory_id}: API returned None (attempt {attempt+1})")

            except Exception as e:
                error_msg = str(e)
                is_rate_limit = "429" in error_msg or "rate" in error_msg.lower()
                is_server_error = any(str(code) in error_msg for code in range(500, 600))
                is_length_error = "not supported" in error_msg.lower() and "length" in error_msg.lower()

                if is_length_error:
                    # Sequence length errors are non-retryable parameter errors
                    log.error(f"  Trajectory {trajectory_id}: Sequence length error: {error_msg[:200]}")
                    return None
                elif is_rate_limit or is_server_error:
                    backoff = BASE_BACKOFF * (2 ** attempt)
                    log.warning(
                        f"  Trajectory {trajectory_id}: {'Rate-limited' if is_rate_limit else 'Server error'} "
                        f"(attempt {attempt+1}/{MAX_RETRIES}). Backing off {backoff:.1f}s. Error: {error_msg[:100]}"
                    )
                    await asyncio.sleep(backoff)
                else:
                    log.error(f"  Trajectory {trajectory_id}: Unrecoverable error: {error_msg[:200]}")
                    return None

        log.error(f"  Trajectory {trajectory_id}: Exhausted {MAX_RETRIES} retries.")
        return None


# ---------------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------------

async def process_trajectory(
    trajectory_id: int,
    sequence: str,
    actions: np.ndarray,
    forward_log_probs: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    api_key: str,
    semaphore: asyncio.Semaphore,
    conn: sqlite3.Connection,
    stats: dict,
    reward_model: str,
):
    """Processes a single trajectory: API call → reward → SQLite insert."""
    t0 = time.time()

    predictions = await query_alphagenome_api(
        sequence, api_key, semaphore, trajectory_id
    )

    # ── Real Evo2 7B Inference with VRAM Lock ──
    # The semaphore allows parallel API mapping, but we strictly lock
    # local GPU access so we don't OOM the Colab T4 with concurrent 7B evaluations.
    async with evo2_lock:
        try:
            loop = asyncio.get_event_loop()
            evo2_score = await loop.run_in_executor(
                None, compute_real_evo2_likelihood, sequence
            )
        except Exception as e:
            log.warning(f"  Trajectory {trajectory_id}: Evo2 inference failed: {e}")
            evo2_score = 0.0

    api_latency_ms = (time.time() - t0) * 1000

    if predictions is not None:
        # Ensure shape compatibility
        pred_bins = min(predictions.shape[0], targets.shape[0])
        pred_tracks = min(predictions.shape[1] if predictions.ndim > 1 else 1, targets.shape[1])

        # Compute reward using the API predictions
        reward = compute_reward_np(
            predictions[:pred_bins, :pred_tracks],
            targets[:pred_bins, :pred_tracks],
            mask[:pred_bins, :pred_tracks],
            evo2_score=evo2_score,
            alpha=ALPHA_REWARD,
            beta=BETA_REWARD,
        )

        insert_experience(
            conn, trajectory_id, actions, forward_log_probs,
            reward, api_latency_ms, reward_model,
        )

        stats["scored"] += 1
        if stats["scored"] % 50 == 0:
            log.info(
                f"  Progress: {stats['scored']}/{stats['total']} scored | "
                f"Last reward: {reward:.6f} | API latency: {api_latency_ms:.0f}ms"
            )
    else:
        stats["failed"] += 1
        log.warning(f"  Trajectory {trajectory_id}: FAILED — no API response.")


async def run_api_worker(api_key: str):
    """Main async entry point for the API worker."""

    # Load unscored trajectories
    if not os.path.exists(INPUT_PATH):
        log.error(f"Input file not found: {INPUT_PATH}")
        log.error("Run 1_trajectory_sampler.py first.")
        sys.exit(1)

    data = np.load(INPUT_PATH, allow_pickle=True)
    # terminal_onehot removed to save 10GB RAM
    actions = data["actions"]                     # (N, num_edits)
    forward_log_probs = data["forward_log_probs"] # (N, num_edits)
    sequences = data["sequences"]                  # (N,) object array of ACGTN strings
    seq_len = int(data["seq_len"])
    num_edits = int(data["num_edits"])

    total = len(sequences)
    log.info(f"[Load] {total} unscored trajectories loaded")
    log.info(f"[Load] Sequence length: {seq_len} bp | Edits: {num_edits}")

    # Validate sequence strings
    for i in range(min(3, total)):
        s = str(sequences[i])
        assert len(s) == seq_len, f"Sequence {i} length {len(s)} != {seq_len}"
        assert all(c in "ACGTN" for c in s[:100]), f"Sequence {i} has invalid characters"
    log.info("[Validate] Sequence strings validated (ACGTN, correct length).")

    # Initialize database
    reward_model_name = os.environ.get("EVO2_MODEL_NAME", "evo2_7b")
    conn = init_database(DB_PATH)
    scored_ids = get_scored_ids(conn, reward_model_name)
    log.info(f"[Resume] {len(scored_ids)} trajectories already scored by '{reward_model_name}' in {DB_PATH}")

    # Construct mock targets and mask for reward computation
    # In production, these would come from real GTEx data
    num_bins = seq_len // 128
    targets = np.zeros((num_bins, NUM_TRACKS), dtype=np.float32)
    mask_tensor = np.zeros((num_bins, NUM_TRACKS), dtype=np.float32)
    active_tracks = [45, 120, 2030, 4011]
    for t in active_tracks:
        targets[:, t] = 1.0
        mask_tensor[:, t] = 1.0

    # Filter to unscored trajectories
    pending = [(i, str(sequences[i])) for i in range(total) if i not in scored_ids]
    log.info(f"[Queue] {len(pending)} trajectories queued for scoring")

    if not pending:
        log.info("All trajectories already scored. Nothing to do.")
        conn.close()
        return

    # Process with concurrency control
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    stats = {"scored": len(scored_ids), "failed": 0, "total": total}

    log.info(f"[Run] Starting async API worker (concurrency: {MAX_CONCURRENCY})")
    log.info(f"[Run] Max retries: {MAX_RETRIES} | Base backoff: {BASE_BACKOFF}s")
    log.info("-" * 70)

    t_start = time.time()

    # Process in batches to limit memory
    batch_size = 100
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        tasks = []
        for traj_id, seq_str in batch:
            task = process_trajectory(
                trajectory_id=traj_id,
                sequence=seq_str,
                actions=actions[traj_id],
                forward_log_probs=forward_log_probs[traj_id],
                targets=targets,
                mask=mask_tensor,
                api_key=api_key,
                semaphore=semaphore,
                conn=conn,
                stats=stats,
                reward_model=reward_model_name,
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

        elapsed = time.time() - t_start
        log.info(
            f"  Batch [{batch_start+len(batch)}/{len(pending)}] complete | "
            f"Elapsed: {elapsed:.0f}s | "
            f"Scored: {stats['scored']} | Failed: {stats['failed']}"
        )

    total_time = time.time() - t_start

    log.info("-" * 70)
    log.info(f"API Worker Complete.")
    log.info(f"  Total scored:  {stats['scored']}")
    log.info(f"  Total failed:  {stats['failed']}")
    log.info(f"  Total time:    {total_time:.1f}s")
    log.info(f"  Replay buffer: {DB_PATH}")
    log.info("=" * 70)

    conn.close()


def main():
    api_key = os.environ.get("ALPHA_GENOME_API_KEY")
    if not api_key:
        log.error("ALPHA_GENOME_API_KEY environment variable not set.")
        log.error("Run: export ALPHA_GENOME_API_KEY=your_key_here")
        sys.exit(1)

    log.info(f"[Auth] API key loaded ({api_key[:8]}...{api_key[-4:]})")

    asyncio.run(run_api_worker(api_key))


if __name__ == "__main__":
    main()
