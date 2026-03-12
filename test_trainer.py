"""
Test script for GFlowNet Trajectory Balance Trainer.

Demonstrates:
  1. Deterministic oracle verification (same input → same output).
  2. Running one full forward trajectory (10 edits) through the MDP.
  3. Computing the terminal reward using deterministic proxy oracles.
  4. Executing jax.value_and_grad and asserting gradient PyTree shape matches
     the Generator's parameter PyTree.
"""

import jax
import jax.numpy as jnp
from jax import tree_util

from gflownet_trainer import (
    init_train_state,
    init_oracle_params,
    training_step,
    compute_reward,
    deterministic_alphagenome_forward,
    deterministic_evo2_prior,
    masked_modality_loss,
)


def assert_pytree_shapes_match(tree_a, tree_b, label_a="A", label_b="B"):
    """Recursively asserts that two PyTrees have identical leaf shapes."""
    leaves_a, struct_a = tree_util.tree_flatten(tree_a)
    leaves_b, struct_b = tree_util.tree_flatten(tree_b)

    assert struct_a == struct_b, (
        f"PyTree structures differ!\n  {label_a}: {struct_a}\n  {label_b}: {struct_b}"
    )

    for i, (la, lb) in enumerate(zip(leaves_a, leaves_b)):
        assert la.shape == lb.shape, (
            f"Leaf {i} shape mismatch: {label_a}={la.shape} vs {label_b}={lb.shape}"
        )


def run_test():
    print("=" * 60)
    print("EEPM3 GFlowNet Trainer Verification Log")
    print("=" * 60)

    # ── Configuration ──────────────────────────────────────────
    seq_len = 1000          # Small sequence for fast test compilation
    num_edits = 10
    num_bins = seq_len // 128  # 7
    num_tracks = 5930
    metadata_dim = 10
    key = jax.random.PRNGKey(42)

    # ── 1. Initialize train state ──────────────────────────────
    print("\n[1] Initializing TrainState (Generator params + log Z)...")
    key, init_key = jax.random.split(key)
    train_state = init_train_state(init_key, seq_len=seq_len, metadata_dim=metadata_dim)

    num_params = sum(p.size for p in tree_util.tree_leaves(train_state.gen_params))
    print(f"    Generator parameters: {num_params:,}")
    print(f"    log Z initialized to: {train_state.log_z:.4f}")

    # ── 2. Initialize frozen oracle parameters ─────────────────
    print("\n[2] Initializing deterministic oracle proxies...")
    key, oracle_key = jax.random.split(key)
    oracle_params = init_oracle_params(oracle_key, num_bins=num_bins, num_tracks=num_tracks)
    print("    AlphaGenome proxy + Evo2 proxy initialized.")

    # ── 3. Verify oracle determinism ───────────────────────────
    print("\n[3] Verifying oracle determinism (same input → same output)...")
    key, wt_key = jax.random.split(key)
    wt_indices = jax.random.randint(wt_key, (seq_len,), 0, 5)
    wt_seq = jax.nn.one_hot(wt_indices, 5, dtype=jnp.float32)

    # Call AlphaGenome proxy twice with identical input
    ag_out_1 = deterministic_alphagenome_forward(wt_seq, oracle_params['alphagenome'],
                                                  num_bins=num_bins, num_tracks=num_tracks)
    ag_out_2 = deterministic_alphagenome_forward(wt_seq, oracle_params['alphagenome'],
                                                  num_bins=num_bins, num_tracks=num_tracks)
    assert jnp.allclose(ag_out_1, ag_out_2), "[FAIL] AlphaGenome proxy is NOT deterministic!"
    assert ag_out_1.shape == (num_bins, num_tracks), f"AG shape mismatch: {ag_out_1.shape}"
    print(f"    [PASS] AlphaGenome proxy deterministic. Output: {ag_out_1.shape}")

    # Call Evo2 proxy twice with identical input
    evo_out_1 = deterministic_evo2_prior(wt_seq, oracle_params['evo2'])
    evo_out_2 = deterministic_evo2_prior(wt_seq, oracle_params['evo2'])
    assert jnp.allclose(evo_out_1, evo_out_2), "[FAIL] Evo2 proxy is NOT deterministic!"
    assert evo_out_1.shape == (), f"Evo2 shape mismatch: {evo_out_1.shape}"
    print(f"    [PASS] Evo2 proxy deterministic. Log-likelihood: {evo_out_1:.6f}")

    # Verify that changing sequence changes output
    modified_seq = wt_seq.at[0].set(jax.nn.one_hot(jnp.int32(3), 5, dtype=jnp.float32))
    ag_modified = deterministic_alphagenome_forward(modified_seq, oracle_params['alphagenome'],
                                                     num_bins=num_bins, num_tracks=num_tracks)
    assert not jnp.allclose(ag_out_1, ag_modified), "[FAIL] Oracle output did not change with different input!"
    print("    [PASS] Different sequences produce different oracle outputs.")

    # ── 4. Create mock inputs ──────────────────────────────────
    print("\n[4] Creating mock wild-type sequence and GTEx targets...")
    target_metadata = jnp.ones((metadata_dim,))

    # Mock GTEx targets and mask (only 3 tracks have data)
    targets = jnp.zeros((num_bins, num_tracks))
    mask = jnp.zeros((num_bins, num_tracks))
    active_tracks = [10, 42, 99]
    for t in active_tracks:
        targets = targets.at[:, t].set(1.0)
        mask = mask.at[:, t].set(1.0)

    print(f"    Sequence shape: {wt_seq.shape}")
    print(f"    Targets shape:  {targets.shape}")
    print(f"    Mask shape:     {mask.shape}")
    print(f"    Active tracks:  {active_tracks}")

    # ── 5. Test masked modality loss ───────────────────────────
    print("\n[5] Testing masked modality loss...")
    l_mask = masked_modality_loss(ag_out_1, targets, mask)
    assert l_mask.shape == (), f"L_mask shape mismatch: {l_mask.shape}"
    print(f"    [PASS] L_mask = {l_mask:.6f} (scalar)")

    # ── 6. Test reward computation ─────────────────────────────
    print("\n[6] Testing reward computation R(x) with deterministic oracles...")
    reward = compute_reward(wt_seq, targets, mask, oracle_params,
                            num_bins=num_bins, num_tracks=num_tracks)
    assert reward.shape == (), f"Reward shape mismatch: {reward.shape}"
    assert reward > 0, f"Reward must be positive, got {reward}"

    # Verify reward is deterministic
    reward2 = compute_reward(wt_seq, targets, mask, oracle_params,
                             num_bins=num_bins, num_tracks=num_tracks)
    assert jnp.allclose(reward, reward2), "[FAIL] Reward is NOT deterministic!"
    print(f"    [PASS] R(x) = {reward:.6f} (positive, deterministic scalar)")

    # ── 7. Full training step with jax.value_and_grad ──────────
    print("\n[7] Executing full training step (10-edit trajectory)...")
    key, step_key = jax.random.split(key)

    loss, grads = training_step(
        train_state, wt_seq, target_metadata, targets, mask,
        oracle_params, step_key,
        seq_len=seq_len, num_edits=num_edits,
    )

    print(f"    TB Loss = {loss:.6f}")
    assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
    print("    [PASS] Loss is finite scalar.")

    # ── 8. Assert gradient PyTree matches parameter PyTree ─────
    print("\n[8] Asserting gradient PyTree shape matches parameter PyTree...")

    assert_pytree_shapes_match(
        grads.gen_params, train_state.gen_params,
        label_a="grad(gen_params)", label_b="gen_params",
    )
    print("    [PASS] grad(gen_params) structure matches gen_params exactly.")

    assert grads.log_z.shape == train_state.log_z.shape, (
        f"log_z grad shape {grads.log_z.shape} != {train_state.log_z.shape}"
    )
    print(f"    [PASS] grad(log_z) shape matches: {grads.log_z.shape}")

    # Verify gradients are non-trivial (not all zeros)
    grad_leaves = tree_util.tree_leaves(grads.gen_params)
    total_grad_norm = sum(float(jnp.sum(g ** 2)) for g in grad_leaves)
    print(f"    Gradient L2² norm (gen_params): {total_grad_norm:.6f}")
    assert total_grad_norm > 0, "Gradients are all zero — no signal is flowing!"
    print("    [PASS] Non-trivial gradients confirmed — signal flows through TB math.")

    # Pretty-print gradient tree structure
    print("\n    Gradient PyTree structure:")
    flat_grads, tree_def = tree_util.tree_flatten_with_path(grads)
    for path, leaf in flat_grads:
        path_str = "/".join(str(p) for p in path)
        print(f"      {path_str:60s} → {leaf.shape}")

    print("\n" + "=" * 60)
    print("All GFlowNet Trainer specifications strictly passed!")
    print("Deterministic oracles verified. Gradients flow correctly.")
    print("=" * 60)


if __name__ == "__main__":
    run_test()
