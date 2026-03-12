import os
import sys
import numpy as np
import pandas as pd
import jax.numpy as jnp
import pyBigWig
from data_pipeline import GTExBigWigParser, GTExDataLoader

def create_mock_environment():
    """Sets up a mock data directory and dummy BigWig files."""
    os.makedirs('data/gtex', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Create mock BigWig files using pyBigWig
    mock_files = [
        ('data/gtex/brain_mock.bw', 10),
        ('data/gtex/liver_mock.bw', 42),
        ('data/gtex/blood_mock.bw', 99)
    ]
    
    records = []
    
    # Example 1Mb interval on chr1: 1000000 to 2048576
    chrom = "chr1"
    
    for filepath, track_idx in mock_files:
        bw = pyBigWig.open(filepath, "w")
        bw.addHeader([(chrom, 3000000)]) # arbitrary header size > 2048576
        
        # Add random mock data within our exact test interval
        # We'll just add intervals every 1000 bases
        starts = np.arange(1000000, 2048576, 1000, dtype=np.int64)
        ends = starts + 500
        vals = np.random.rand(len(starts)).astype(np.float64)
        
        # pyBigWig requires lists or arrays of equal length for chromosomes, starts, ends, values
        chroms = [chrom] * len(starts)
        bw.addEntries(chroms, starts.tolist(), ends=ends.tolist(), values=vals.tolist())
        bw.close()
        
        # Add to mock metadata
        records.append({
            'filepath': filepath,
            'tissue': 'MockTissue',
            'age_bracket': '20-30',
            'track_index': track_idx
        })
        
    df = pd.DataFrame(records)
    df.to_csv('data/metadata.csv', index=False)
    return df

def test_pipeline():
    print("--------------------------------------------------")
    print("EEPM3 Data Pipeline Verification Log")
    print("--------------------------------------------------")
    print("Initializing Mock Environment...")
    meta_df = create_mock_environment()
    
    print("Setting up GTExParser and DataLoader...")
    parser = GTExBigWigParser(data_dir="data/gtex", metadata_path="data/metadata.csv", seq_len=1048576, bin_size=128, target_bins=7812)
    loader = GTExDataLoader(parser=parser, num_tracks=5930)
    
    # Batch size of 2
    # Sample 0 has brain and liver available.
    sample_0_meta = meta_df[meta_df['track_index'].isin([10, 42])]
    # Sample 1 has only blood available.
    sample_1_meta = meta_df[meta_df['track_index'].isin([99])]
    
    batch_metadata = [sample_0_meta, sample_1_meta]
    
    print("Generating training batch (B=2)...")
    # interval starting at 1000000
    batch = loader.generate_batch(batch_metadata, chrom="chr1", start=1000000)
    
    T = batch['targets']
    M = batch['masks']
    
    print("\n[Executing Required Assertions]")
    # 1. Shape Test
    assert T.shape == (2, 7812, 5930), f"Expected shape (2, 7812, 5930), got {T.shape}"
    assert M.shape == (2, 7812, 5930), f"Expected M shape (2, 7812, 5930), got {M.shape}"
    print("[PASS] Tensor T and M shapes are strictly (B, 7812, 5930).")
    
    # 2. Mask Contents Test
    unique_vals = jnp.unique(M)
    assert jnp.all(jnp.isin(unique_vals, jnp.array([0.0, 1.0]))), f"Mask contains non-binary values: {unique_vals}"
    print("[PASS] Mask M correctly contains only 0s and 1s.")
    
    # 3. Missing Data Masking Test
    # T * (1 - M) should yield all zeros
    unmasked_leakage = jnp.sum(T * (1 - M))
    assert unmasked_leakage == 0.0, f"Found unmasked data in T! Leakage sum: {unmasked_leakage}"
    print("[PASS] T * (1 - M) strictly yields zeros. Missing data is perfectly masked.")
    
    # Check that tracks WITH data are populated properly
    assert jnp.sum(T[0, :, 10]) > 0.0, "[FAIL] Expected mock data on Sample 0, Track 10"
    assert jnp.sum(M[0, :, 10]) == 7812.0, "[FAIL] Expected mask to be fully active for Track 10"
    assert jnp.sum(T[1, :, 99]) > 0.0, "[FAIL] Expected mock data on Sample 1, Track 99"
    assert jnp.sum(M[1, :, 99]) == 7812.0, "[FAIL] Expected mask to be fully active for Track 99"
    
    print("\n--------------------------------------------------")
    print("All pipeline specifications strictly passed!")
    print("PyTree output valid for Flambax/JAX masking.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    test_pipeline()
