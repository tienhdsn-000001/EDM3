"""
GFlowNet Trajectory Balance Trainer for EEPM3.

Weaves together:
  - Agent A's data pipeline outputs (targets T, masks M)
  - Agent B's MDP environment (GFlowNetEnv, GeneratorPolicy)
  - Deterministic proxy oracles for AlphaGenome and Evo2
  - Trajectory Balance (TB) Loss with learnable partition function Z

All functions are pure and jax.jit-compatible.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import NamedTuple, Dict, Any, Tuple

from gflownet_env import GFlowNetEnv, EnvState, GeneratorPolicy


# ---------------------------------------------------------------------------
# Deterministic Proxy Oracles (Frozen Foundation Model Surrogates)
# ---------------------------------------------------------------------------

class AlphaGenomeProxy(nn.Module):
    """
    Deterministic proxy for AlphaGenome.

    A frozen single-layer linear projection that maps a sequence summary
    to a (num_bins, num_tracks) prediction tensor. Same input always produces
    the same output. Changing one base changes the output deterministically.

    This is NOT AlphaGenome — it is a fixed landscape that the GFlowNet can
    learn to optimize against, proving the training loop's capacity to converge.
    """
    num_bins: int = 781
    num_tracks: int = 5930

    @nn.compact
    def __call__(self, seq_summary: jnp.ndarray) -> jnp.ndarray:
        """
        seq_summary: (vocab_size,) — mean base composition of the sequence.
        Returns: (num_bins, num_tracks)
        """
        # Project through a frozen dense layer to a flat output
        # Keep output small enough to avoid OOM: project to num_bins, then tile
        projection = nn.Dense(self.num_bins, name="ag_proj")(seq_summary)  # (num_bins,)
        # Tile across tracks with a learned scale vector
        track_scale = nn.Dense(self.num_tracks, name="ag_track_scale")(seq_summary)  # (num_tracks,)
        # Outer product: (num_bins, 1) * (1, num_tracks) → (num_bins, num_tracks)
        return projection[:, None] * track_scale[None, :]


class Evo2Proxy(nn.Module):
    """
    Deterministic proxy for Evo2 sequence plausibility.

    A frozen two-layer MLP that maps sequence composition to a scalar
    log-likelihood score. Same input → same output, always.
    """
    @nn.compact
    def __call__(self, seq_summary: jnp.ndarray) -> jnp.ndarray:
        """
        seq_summary: (vocab_size,) — mean base composition.
        Returns: scalar log-likelihood.
        """
        x = nn.Dense(16, name="evo_h1")(seq_summary)
        x = jax.nn.tanh(x)
        x = nn.Dense(1, name="evo_h2")(x)
        return x.squeeze(-1)  # scalar


def init_oracle_params(key: jnp.ndarray, vocab_size: int = 5,
                       num_bins: int = 781, num_tracks: int = 5930) -> Dict[str, Any]:
    """
    Initialize frozen oracle parameters with a fixed seed.
    These parameters are never updated during training.
    """
    key_ag, key_evo = jax.random.split(key)

    ag_proxy = AlphaGenomeProxy(num_bins=num_bins, num_tracks=num_tracks)
    ag_params = ag_proxy.init(key_ag, jnp.zeros((vocab_size,)))

    evo_proxy = Evo2Proxy()
    evo_params = evo_proxy.init(key_evo, jnp.zeros((vocab_size,)))

    return {'alphagenome': ag_params, 'evo2': evo_params}


def deterministic_alphagenome_forward(
    sequence: jnp.ndarray,
    oracle_params: Dict,
    num_bins: int = 781,
    num_tracks: int = 5930,
) -> jnp.ndarray:
    """
    Deterministic AlphaGenome proxy forward pass.

    sequence: (L, 5) one-hot DNA sequence
    oracle_params: frozen AlphaGenomeProxy parameters

    Returns: (num_bins, num_tracks) — same input always produces same output.
    """
    seq_summary = jnp.mean(sequence, axis=0)  # (5,) — base composition
    proxy = AlphaGenomeProxy(num_bins=num_bins, num_tracks=num_tracks)
    return proxy.apply(oracle_params, seq_summary)


def deterministic_evo2_prior(
    sequence: jnp.ndarray,
    oracle_params: Dict,
) -> jnp.ndarray:
    """
    Deterministic Evo2 proxy forward pass.

    sequence: (L, 5) one-hot DNA sequence
    oracle_params: frozen Evo2Proxy parameters

    Returns: scalar log-likelihood — same input always produces same output.
    """
    seq_summary = jnp.mean(sequence, axis=0)  # (5,)
    proxy = Evo2Proxy()
    return proxy.apply(oracle_params, seq_summary)


# ---------------------------------------------------------------------------
# Masked Modality Loss
# ---------------------------------------------------------------------------

def masked_modality_loss(predictions: jnp.ndarray, targets: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """
    Computes L_mask = mean of squared errors only where mask == 1.

    predictions: (num_bins, num_tracks) from AlphaGenome
    targets:     (num_bins, num_tracks) from GTEx pipeline
    mask:        (num_bins, num_tracks) binary availability mask

    Returns scalar loss.
    """
    sq_error = (predictions - targets) ** 2
    masked_error = sq_error * mask

    # Avoid division by zero if mask is entirely empty
    num_valid = jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.sum(masked_error) / num_valid


# ---------------------------------------------------------------------------
# Reward Function (from PRD)
# ---------------------------------------------------------------------------

def compute_reward(
    terminal_seq: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    oracle_params: Dict,
    alpha: float = 1.0,
    beta: float = 0.1,
    num_bins: int = 781,
    num_tracks: int = 5930,
) -> jnp.ndarray:
    """
    R(x) = exp(-α · L_mask(AG(x), T, M)) + β · log P_Evo(x)

    terminal_seq: (L, 5) one-hot final mutated sequence
    targets:      (num_bins, num_tracks) demographic target tensor
    mask:         (num_bins, num_tracks) binary mask
    oracle_params: frozen oracle parameter dict

    Returns scalar reward.
    """
    ag_pred = deterministic_alphagenome_forward(
        terminal_seq, oracle_params['alphagenome'],
        num_bins=num_bins, num_tracks=num_tracks,
    )
    l_mask = masked_modality_loss(ag_pred, targets, mask)

    log_p_evo = deterministic_evo2_prior(terminal_seq, oracle_params['evo2'])

    reward = jnp.exp(-alpha * l_mask) + beta * log_p_evo
    # Clamp reward to be strictly positive (required for log in TB loss)
    reward = jnp.maximum(reward, 1e-8)
    return reward


# ---------------------------------------------------------------------------
# Trajectory Balance (TB) Loss
# ---------------------------------------------------------------------------

def compute_backward_log_prob(num_edits_remaining: jnp.ndarray) -> jnp.ndarray:
    """
    Backward policy P_B is uniform: probability of undoing any specific edit
    is 1/N where N is the number of edits remaining to undo.

    Returns log(1/N) = -log(N).
    """
    return -jnp.log(jnp.maximum(num_edits_remaining, 1.0))


def tb_loss(
    log_z: jnp.ndarray,
    forward_log_probs: jnp.ndarray,
    log_reward: jnp.ndarray,
    num_edits: int,
) -> jnp.ndarray:
    """
    Trajectory Balance Loss:
      L_TB = (log Z + Σ log P_F(a|s) - log R(x) - Σ log P_B(a|s'))²

    log_z:              Learnable scalar (partition function estimate)
    forward_log_probs:  (T,) array of log P_F for each action taken
    log_reward:         Scalar log R(x) of terminal state
    num_edits:          Total number of mutation edits in trajectory

    Returns scalar TB loss.
    """
    # Forward flow: log Z + sum of forward log-probs
    forward_flow = log_z + jnp.sum(forward_log_probs)

    # Backward flow: log R(x) + sum of backward log-probs
    # P_B is uniform: at step i (undoing), there are (num_edits - i) remaining
    # So log P_B at step i = -log(num_edits - i)
    edit_indices = jnp.arange(num_edits)
    remaining_at_step = num_edits - edit_indices  # N, N-1, ..., 1
    backward_log_probs = compute_backward_log_prob(remaining_at_step.astype(jnp.float32))
    backward_flow = log_reward + jnp.sum(backward_log_probs)

    return (forward_flow - backward_flow) ** 2


# ---------------------------------------------------------------------------
# TrainState container for learnable parameters
# ---------------------------------------------------------------------------

class TrainState(NamedTuple):
    gen_params: Any       # Generator (P_F) Flax parameter tree
    log_z: jnp.ndarray    # Learnable scalar log-partition function


def init_train_state(
    key: jnp.ndarray,
    seq_len: int,
    metadata_dim: int = 10,
) -> TrainState:
    """
    Initializes the generator parameters and log Z.
    """
    policy = GeneratorPolicy(seq_len=seq_len)

    # Dummy inputs for parameter initialization (batch size 1)
    # 6 channels: 5 one-hot DNA + 1 mutated mask
    dummy_seq = jnp.zeros((1, seq_len, 6))
    dummy_meta = jnp.zeros((1, metadata_dim))

    gen_params = policy.init(key, dummy_seq, dummy_meta)
    log_z = jnp.float32(0.0)

    return TrainState(gen_params=gen_params, log_z=log_z)


# ---------------------------------------------------------------------------
# Full Differentiable Training Step
# ---------------------------------------------------------------------------

def run_trajectory_and_compute_loss(
    train_state: TrainState,
    wt_seq: jnp.ndarray,
    target_metadata: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    oracle_params: Dict,
    key: jnp.ndarray,
    seq_len: int,
    num_edits: int = 10,
    alpha: float = 1.0,
    beta: float = 0.1,
) -> jnp.ndarray:
    """
    Runs a full forward trajectory through the MDP, computes the terminal
    reward, and returns the Trajectory Balance loss.

    This entire function is pure and differentiable w.r.t. train_state.
    oracle_params are frozen and not part of the gradient computation.
    """
    env = GFlowNetEnv(seq_len=seq_len, max_edits=num_edits)
    policy = GeneratorPolicy(seq_len=seq_len)
    state = env.reset(wt_seq)

    def scan_step(carry, step_key):
        current_state, forward_lps = carry

        # Get valid action mask
        valid_mask = env.get_valid_actions(current_state)

        # Build 6-channel input: (L, 5) one-hot + (L, 1) mutated mask
        mutated_channel = current_state.mutated.astype(jnp.float32)[:, None]  # (L, 1)
        state_6ch = jnp.concatenate([current_state.seq, mutated_channel], axis=-1)  # (L, 6)

        # Run forward policy (unbatched → add batch dim)
        seq_batched = state_6ch[None, ...]                # (1, L, 6)
        meta_batched = target_metadata[None, ...]         # (1, M)
        raw_logits = policy.apply(train_state.gen_params, seq_batched, meta_batched)
        raw_logits = raw_logits[0]                        # (action_space,)

        # Mask invalid actions and block terminate during trajectory
        masked_logits = jnp.where(valid_mask, raw_logits, -1e9)
        masked_logits = masked_logits.at[env.terminate_action].set(-1e9)

        # Sample action
        action = jax.random.categorical(step_key, masked_logits)

        # Compute log P_F(a|s) using log-softmax of the masked logits
        log_probs = jax.nn.log_softmax(masked_logits)
        log_p_f = log_probs[action]

        # Step environment
        next_state, _ = env.step(current_state, action)

        # Accumulate forward log-probs in a fixed-size array
        return (next_state, forward_lps), log_p_f

    # Pre-allocate carry
    init_forward_lps = jnp.zeros(())  # placeholder, not used in carry
    keys = jax.random.split(key, num_edits + 1)
    trajectory_keys = keys[:num_edits]
    reward_key = keys[num_edits]

    (final_state, _), forward_log_probs = jax.lax.scan(
        scan_step,
        (state, init_forward_lps),
        trajectory_keys,
    )

    # Compute terminal reward using deterministic oracles
    num_bins = seq_len // 128
    reward = compute_reward(
        final_state.seq, targets, mask, oracle_params,
        alpha=alpha, beta=beta,
        num_bins=num_bins, num_tracks=targets.shape[-1],
    )
    log_reward = jnp.log(jnp.maximum(reward, 1e-8))

    # Compute TB loss
    loss = tb_loss(train_state.log_z, forward_log_probs, log_reward, num_edits)
    return loss


def training_step(
    train_state: TrainState,
    wt_seq: jnp.ndarray,
    target_metadata: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
    oracle_params: Dict,
    key: jnp.ndarray,
    seq_len: int,
    num_edits: int = 10,
) -> tuple:
    """
    Single pure training step: computes loss AND gradients w.r.t. train_state.
    Returns (loss, grads) where grads has the same PyTree structure as train_state.
    oracle_params are frozen and do not receive gradients.
    """
    loss, grads = jax.value_and_grad(run_trajectory_and_compute_loss)(
        train_state, wt_seq, target_metadata, targets, mask,
        oracle_params, key, seq_len, num_edits,
    )
    return loss, grads
