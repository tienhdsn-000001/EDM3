"""
Phase 5, Script 1: Retrospective Backward Synthesis (RBS) Data Augmenter.

Identifies the top-10% highest-reward experiences from the experience replay buffer,
then hallucates 5-10 valid alternative forward trajectories that arrive at the exact same
terminal sequence. This multiplies the high-value training signal.

RBS Logic:
  Given terminal sequence T that required mutations at positions {p1, p2, ..., p_k}:
  1. Any permutation of applying those same (position, base) mutations is a valid
     alternative forward trajectory.
  2. We sample new orderings, compute log P_B for each, and store them as
     augmented training examples with the same R(x).

This script reads from data/experience_replay.db and writes to
data/experience_replay_augmented.db.
"""

import os
import sys
import sqlite3
import numpy as np
import itertools
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_DB = "data/experience_replay.db"
AUGMENTED_DB = "data/experience_replay_augmented.db"
TOP_PERCENTILE = 0.10       # Top 10% by reward
HALLUCINATIONS_PER_TRAJ = 5  # New trajectories per top sequence
SEQ_LEN = 100_000
NUM_EDITS = 10
VOCAB_SIZE = 5


# ---------------------------------------------------------------------------
# Database I/O
# ---------------------------------------------------------------------------

def init_augmented_db(db_path: str) -> sqlite3.Connection:
    """Creates the augmented experience database."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_trajectory_id INTEGER,
            actions BLOB NOT NULL,
            forward_log_probs BLOB NOT NULL,
            reward REAL NOT NULL,
            is_augmented INTEGER DEFAULT 0,
            augmentation_method TEXT
        )
    """)
    conn.commit()
    return conn


def load_source_experiences(db_path: str) -> list:
    """Loads all experiences from the source database."""
    if not os.path.exists(db_path):
        print(f"[ERROR] Source database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT trajectory_id, actions, forward_log_probs, reward FROM experiences ORDER BY reward DESC"
    )

    experiences = []
    for row in cursor:
        traj_id, actions_bytes, lp_bytes, reward = row
        actions = np.frombuffer(actions_bytes, dtype=np.int32).copy()
        forward_log_probs = np.frombuffer(lp_bytes, dtype=np.float32).copy()

        experiences.append({
            "trajectory_id": traj_id,
            "actions": actions,
            "forward_log_probs": forward_log_probs,
            "reward": reward,
        })

    conn.close()
    return experiences


# ---------------------------------------------------------------------------
# RBS Core: Backward Trajectory Hallucination
# ---------------------------------------------------------------------------

def extract_mutations_from_actions(actions: np.ndarray) -> List[Tuple[int, int]]:
    """
    Extracts (position, base) mutations from action indices.
    Action encoding: action = position * VOCAB_SIZE + base
    """
    mutations = []
    for a in actions:
        a = int(a)
        if a == SEQ_LEN * VOCAB_SIZE:  # terminate action
            continue
        pos = a // VOCAB_SIZE
        base = a % VOCAB_SIZE
        mutations.append((pos, base))
    return mutations


def synthesize_alternative_trajectory(
    mutations: List[Tuple[int, int]],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given the set of mutations that produce a terminal state, generate
    an alternative forward trajectory by randomly permuting the order of
    mutations. This is valid because mutations are at distinct positions
    (enforced by the action mask).

    Returns: (permuted_actions, synthetic_forward_log_probs)
    """
    num_muts = len(mutations)

    # Random permutation of mutation order
    perm = rng.permutation(num_muts)
    permuted_mutations = [mutations[i] for i in perm]

    # Convert back to action indices
    permuted_actions = np.array(
        [pos * VOCAB_SIZE + base for pos, base in permuted_mutations],
        dtype=np.int32,
    )

    # Pad to NUM_EDITS if fewer mutations than edits
    if len(permuted_actions) < NUM_EDITS:
        # Fill with terminate actions
        pad = np.full(NUM_EDITS - len(permuted_actions),
                      SEQ_LEN * VOCAB_SIZE, dtype=np.int32)
        permuted_actions = np.concatenate([permuted_actions, pad])

    # Synthetic forward log-probs: uniform assumption over valid actions at each step
    # At step i, there are approximately (SEQ_LEN - i) * 4 valid mutation actions
    # (each position has 4 novel bases, minus already-mutated positions)
    synthetic_log_probs = np.zeros(NUM_EDITS, dtype=np.float32)
    for i in range(min(num_muts, NUM_EDITS)):
        num_valid_positions = SEQ_LEN - i
        num_valid_actions = num_valid_positions * (VOCAB_SIZE - 1)  # 4 valid bases per position
        synthetic_log_probs[i] = -np.log(max(num_valid_actions, 1))

    return permuted_actions, synthetic_log_probs


def hallucinate_trajectories(
    experience: dict,
    num_hallucinations: int,
    rng: np.random.Generator,
) -> list:
    """
    Given a high-reward experience, generate num_hallucinations alternative
    forward trajectories that arrive at the same terminal state.
    """
    mutations = extract_mutations_from_actions(experience["actions"])

    if len(mutations) < 2:
        # Need at least 2 mutations to permute
        return []

    hallucinated = []
    seen_orderings = set()

    for _ in range(num_hallucinations * 3):  # Over-sample to find unique permutations
        if len(hallucinated) >= num_hallucinations:
            break

        perm_actions, perm_log_probs = synthesize_alternative_trajectory(mutations, rng)

        # Deduplicate by action ordering
        ordering_key = tuple(perm_actions[:len(mutations)])
        if ordering_key in seen_orderings:
            continue
        seen_orderings.add(ordering_key)

        # Skip if identical to original
        if np.array_equal(perm_actions, experience["actions"]):
            continue

        hallucinated.append({
            "source_trajectory_id": experience["trajectory_id"],
            "actions": perm_actions,
            "forward_log_probs": perm_log_probs,
            "reward": experience["reward"],
        })

    return hallucinated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("EEPM3 Phase 5: Retrospective Backward Synthesis (RBS) Augmenter")
    print("=" * 70)

    # Load source data
    print(f"[Load] Reading source database: {SOURCE_DB}")
    experiences = load_source_experiences(SOURCE_DB)
    total = len(experiences)

    if total == 0:
        print("[ERROR] No experiences found in source database.")
        print("  Run 2_api_worker.py first to populate the experience replay buffer.")
        sys.exit(1)

    print(f"[Load] {total} experiences loaded.")

    # Identify top percentile
    rewards = np.array([e["reward"] for e in experiences])
    cutoff_idx = max(1, int(total * TOP_PERCENTILE))
    cutoff_reward = rewards[cutoff_idx - 1] if total > 0 else 0.0

    top_experiences = experiences[:cutoff_idx]
    print(f"\n[Select] Top {TOP_PERCENTILE*100:.0f}% = {cutoff_idx} experiences")
    print(f"  Reward cutoff: {cutoff_reward:.6f}")
    print(f"  Top reward range: [{rewards[0]:.6f}, {rewards[cutoff_idx-1]:.6f}]")

    # Initialize augmented database
    print(f"\n[Init] Creating augmented database: {AUGMENTED_DB}")
    os.makedirs(os.path.dirname(AUGMENTED_DB) if os.path.dirname(AUGMENTED_DB) else ".", exist_ok=True)
    aug_conn = init_augmented_db(AUGMENTED_DB)

    # First: copy ALL original experiences to augmented DB
    print(f"[Copy] Copying {total} original experiences to augmented database...")
    for exp in experiences:
        aug_conn.execute(
            "INSERT INTO experiences "
            "VALUES (NULL, ?, ?, ?, ?, 0, 'original')",
            (
                exp["trajectory_id"],
                exp["actions"].tobytes(),
                exp["forward_log_probs"].tobytes(),
                exp["reward"],
            ),
        )
    aug_conn.commit()

    # RBS Hallucination
    print(f"\n[RBS] Hallucinating {HALLUCINATIONS_PER_TRAJ} trajectories per top experience...")
    rng = np.random.default_rng(seed=42)

    total_augmented = 0
    for i, exp in enumerate(top_experiences):
        hallucinated = hallucinate_trajectories(exp, HALLUCINATIONS_PER_TRAJ, rng)

        for h in hallucinated:
            aug_conn.execute(
                "INSERT INTO experiences "
                "VALUES (NULL, ?, ?, ?, ?, 1, 'rbs_permutation')",
                (
                    h["source_trajectory_id"],
                    h["actions"].tobytes(),
                    h["forward_log_probs"].tobytes(),
                    h["reward"],
                ),
            )
            total_augmented += 1

        if (i + 1) % max(1, cutoff_idx // 10) == 0:
            print(f"  [{i+1}/{cutoff_idx}] Augmented: {total_augmented} trajectories")
            aug_conn.commit()

    aug_conn.commit()

    # Summary
    orig_count = aug_conn.execute("SELECT COUNT(*) FROM experiences WHERE is_augmented=0").fetchone()[0]
    aug_count = aug_conn.execute("SELECT COUNT(*) FROM experiences WHERE is_augmented=1").fetchone()[0]
    total_count = aug_conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]

    print(f"\n[Summary]")
    print(f"  Original experiences:  {orig_count}")
    print(f"  RBS augmented:         {aug_count}")
    print(f"  Total experiences:     {total_count}")
    print(f"  Augmentation ratio:    {total_count / max(orig_count, 1):.1f}x")

    # Trace log: show one hallucination example
    print(f"\n[Trace] Example RBS hallucination:")
    if top_experiences:
        example = top_experiences[0]
        mutations = extract_mutations_from_actions(example["actions"])
        print(f"  Source trajectory {example['trajectory_id']} (reward={example['reward']:.6f})")
        print(f"  Original mutations: {mutations[:5]}...")
        print(f"  Original actions:   {example['actions'][:5]}...")

        hallucinated = hallucinate_trajectories(example, 2, rng)
        for j, h in enumerate(hallucinated):
            h_mutations = extract_mutations_from_actions(h["actions"])
            print(f"  Hallucination {j+1}: {h_mutations[:5]}...")
            print(f"    Actions: {h['actions'][:5]}... (same terminal state, different order)")

    aug_conn.close()

    print(f"\n[Output] Augmented database: {AUGMENTED_DB}")
    print("=" * 70)


if __name__ == "__main__":
    main()
