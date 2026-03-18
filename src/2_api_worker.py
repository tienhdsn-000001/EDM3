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
import gc
import json
import time
import sqlite3
import asyncio
import logging
import torch
import numpy as np
from typing import Optional

# --- Biological Prior: Nucleotide Transformer (T4 Compatible) ---
from transformers import AutoTokenizer, AutoModelForMaskedLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("api_worker")

# ── Pre-emptive VRAM Clearance (Step 1 JAX/XLA leftovers) ──
if torch.cuda.is_available():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # Log initial state
    _total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    _free = torch.cuda.mem_get_info()[0] / (1024**3)
    log.info(f"[Memory] System Startup | Free: {_free:.2f}GB / {_total:.2f}GB")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_PATH = "data/unscored_trajectories.npz"
DB_PATH = "data/experience_replay.db"
MAX_CONCURRENCY = 2          # NT 500M is lightweight enough for concurrency 2 on T4
MAX_RETRIES = 12             # More retries for long overnight runs
BASE_BACKOFF = 3.0           # Increased base backoff for gRPC stability
ALPHA_REWARD = 1.0           # α in R(x) = exp(-α·L_mask) + β·log P_Prior
BETA_REWARD = 0.1            # β weight on Bio-Prior term
PRIOR_MODEL_NAME = "InstaDeepAI/nucleotide-transformer-500m-human-ref"

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
    prior_score: float = 0.0,
    alpha: float = 1.0,
    beta: float = 0.1,
) -> float:
    """
    R(x) = exp(-α · L_mask + β · PriorScore)
    Standard exponential formulation for biological rewards.
    """
    l_mask = masked_modality_loss_np(ag_predictions, targets, mask)
    # We combine the penalties in the exponent to preserve gradient variance
    reward = np.exp(-alpha * l_mask + beta * prior_score)
    return float(max(reward, 1e-12))


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
# Biological Prior: Nucleotide Transformer
# ---------------------------------------------------------------------------

_prior_model = None
_prior_tokenizer = None
prior_lock = asyncio.Lock()

def _log_gpu_memory(label: str):
    """Logs the current GPU memory usage."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    log.info(f"[Memory] {label} | Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")

def _get_prior_resources():
    """Loads the Nucleotide Transformer 500M. Prefers GPU but falls back to CPU if memory is tight."""
    global _prior_model, _prior_tokenizer
    
    # Check if GPU is already saturated (T4 usually 14-15GB capacity)
    gpu_full = False
    if torch.cuda.is_available():
        _log_gpu_memory("Pre-load Check")
        free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
        if free_mem < 2.5: # We need roughly 2.5GB for weights + activations
            log.warning(f"[Prior] GPU memory tight ({free_mem:.2f}GB free). Forcing CPU for Prior model.")
            gpu_full = True
            
    device = "cuda" if (torch.cuda.is_available() and not gpu_full) else "cpu"
    
    if _prior_model is None:
        try:
            log.info(f"[Prior] Loading Nucleotide Transformer: {PRIOR_MODEL_NAME} -> {device}")
            _prior_tokenizer = AutoTokenizer.from_pretrained(PRIOR_MODEL_NAME)
            _prior_model = AutoModelForMaskedLM.from_pretrained(
                PRIOR_MODEL_NAME,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True
            ).to(device)
            _prior_model.eval()
            log.info(f"[Prior] Nucleotide Transformer is LIVE on {device}.")
            if device == "cuda":
                _log_gpu_memory("Post-load Check")
        except Exception as e:
            log.error(f"[Prior] Failed to load model: {e}")
            raise
    return _prior_model, _prior_tokenizer


@torch.no_grad()
async def compute_biological_prior_score(sequence: str) -> float:
    """
    Paranoid Scoring Mode:
    1. Tokenize the ENTIRE sequence on CPU.
    2. Split into 1024-token chunks (the model's hard context limit).
    3. Run forward passes one-by-one on GPU.
    4. Aggressively del and empty_cache to prevent VRAM accumulation.
    """
    async with prior_lock:
        model, tokenizer = _get_prior_resources()
        device = next(model.parameters()).device
        
        # Tokenize full sequence on CPU (avoid device=device here)
        # Note: 131kb sequence results in ~22k tokens.
        full_inputs = tokenizer(
            sequence, 
            return_tensors="pt", 
            padding=False, 
            truncation=False
        )
        
        input_ids = full_inputs["input_ids"][0] # Shape: [N_total]
        attention_mask = full_inputs["attention_mask"][0]
        
        N_total = input_ids.shape[0]
        chunk_size = 1024 # Model context limit
        
        all_losses = []
        
        # Sequentially process windows of tokens instead of BP windows
        for start_idx in range(0, N_total, chunk_size):
            end_idx = min(start_idx + chunk_size, N_total)
            
            # Prepare single chunk
            chunk_input_ids = input_ids[start_idx:end_idx].unsqueeze(0).to(device)
            chunk_mask = attention_mask[start_idx:end_idx].unsqueeze(0).to(device)
            
            try:
                # Forward pass
                outputs = model(
                    input_ids=chunk_input_ids,
                    attention_mask=chunk_mask,
                    labels=chunk_input_ids # CE Loss relative to original tokens
                )
                
                loss_val = float(outputs.loss.cpu().item())
                all_losses.append(loss_val)
                
                # Aggressive Cleanup
                del outputs
                del chunk_input_ids
                del chunk_mask
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError:
                log.warning(f"[Memory] OOM during chunk {start_idx}:{end_idx}. Skipping chunk.")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                log.error(f"[Prior] Error scoring chunk: {e}")
                raise
        
        # Free CPU tokens
        del input_ids
        del attention_mask
        del full_inputs
        
        # Negative loss = log-likelihood
        return -float(np.mean(all_losses)) if all_losses else 0.0


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

    api_start = time.time()
    predictions = await query_alphagenome_api(
        sequence, api_key, semaphore, trajectory_id
    )
    api_time = (time.time() - api_start) * 1000
    
    if predictions is None:
        stats["failed"] += 1
        log.warning(f"  Trajectory {trajectory_id}: FAILED — No AlphaGenome API response.")
        return

    # ── Foundation Model Prior (Nucleotide Transformer) ──
    gpu_start = time.time()
    try:
        prior_score = await compute_biological_prior_score(sequence)
    except Exception as e:
        stats["failed"] += 1
        log.warning(f"  Trajectory {trajectory_id}: FAILED — Prior scoring error: {e}")
        return
    gpu_time = (time.time() - gpu_start) * 1000

    total_latency_ms = (time.time() - t0) * 1000

    # Ensure shape compatibility
    pred_bins = min(predictions.shape[0], targets.shape[0])
    pred_tracks = min(predictions.shape[1] if predictions.ndim > 1 else 1, targets.shape[1])

    # Compute reward using the API predictions
    reward = compute_reward_np(
        predictions[:pred_bins, :pred_tracks],
        targets[:pred_bins, :pred_tracks],
        mask[:pred_bins, :pred_tracks],
        prior_score=prior_score,
        alpha=ALPHA_REWARD,
        beta=BETA_REWARD,
    )

    insert_experience(
        conn, trajectory_id, actions, forward_log_probs,
        reward, total_latency_ms, reward_model,
    )

    stats["scored"] += 1
    if stats["scored"] % 25 == 0:
        log.info(
            f"  Progress: {stats['scored']}/{stats['total']} scored | "
            f"Reward: {reward:.6f} | API: {api_time/1000:.1f}s | GPU: {gpu_time/1000:.1f}s"
        )


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
    reward_model_name = "nucleotide_transformer_500m"

    # Set PyTorch memory allocator settings to reduce fragmentation
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    conn = init_database(DB_PATH)

    # ── Dead Score Purger & Force Override ──
    # 1. Manual Force Override
    if os.environ.get("FORCE_RESCORE") == "1":
        log.warning(f"[Force] Wiping scores for {reward_model_name} as requested.")
        conn.execute("DELETE FROM experiences WHERE reward_model = ?", (reward_model_name,))
        conn.commit()

    # 2. Automated Dead-Score Check
    # Validate the quality of existing scores. If standard deviation is low, 
    # the previous run likely failed to load weights or used a placeholder.
    cursor = conn.execute(
        "SELECT reward FROM experiences WHERE reward_model = ? LIMIT 100", 
        (reward_model_name,)
    )
    existing_rewards = [row[0] for row in cursor.fetchall()]
    
    # Increased sensitivity (threshold 1e-4). Authentic biological rewards 
    # should have significant variance.
    if len(existing_rewards) >= 10 and np.std(existing_rewards) < 1e-4:
        log.warning("=" * 70)
        log.warning(f"[Purge] Detected 'Low Variance' scores for {reward_model_name}.")
        log.warning("[Purge] Previous run likely used a fallback or placeholder.")
        log.warning("[Purge] Wiping broken scores to restore experimental rigor...")
        log.warning("=" * 70)
        conn.execute("DELETE FROM experiences WHERE reward_model = ?", (reward_model_name,))
        conn.commit()

    scored_ids = get_scored_ids(conn, reward_model_name)
    log.info(f"[Resume] {len(scored_ids)} valid trajectories already scored by '{reward_model_name}'")

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
        
        # Explicitly clear cache after each batch to reclaim activation memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
