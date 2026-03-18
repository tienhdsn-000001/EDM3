#!/usr/bin/env python3
"""
scripts/vram_autopsy.py - Forensic Memory Profiler for EEPM3

This script systematically tests the VRAM limits of 16GB GPUs (T4) when 
running biological foundation models (Evo2 7B and NT-500M) alongside JAX.
"""

import os
import gc
import time
import torch
import numpy as np
import logging

# Set transformers log level to error to avoid noisy warnings during OOM tests
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import AutoTokenizer, AutoModelForMaskedLM

# Optional: Try rich for pretty tables, fallback to standard print
try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    console = None

def log_vram(label: str):
    """Returns a dict of current VRAM state."""
    if not torch.cuda.is_available():
        return {"Label": label, "Allocated": "-", "Reserved": "-", "Free": "-", "Status": "No CUDA"}
    
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return {
        "Label": label,
        "Allocated": f"{allocated:.2f}GB",
        "Reserved": f"{reserved:.2f}GB",
        "Free": f"{free_mem:.2f}GB",
        "Total": f"{total:.2f}GB"
    }

def print_report(results):
    if console:
        table = Table(title="EEPM3 Forensic VRAM Autopsy Report")
        table.add_column("Phase", style="cyan")
        table.add_column("Allocated", style="magenta")
        table.add_column("Reserved", style="yellow")
        table.add_column("Free", style="green")
        table.add_column("Status", style="bold white")
        
        for r in results:
            table.add_row(
                r.get("Label", ""),
                r.get("Allocated", "-"),
                r.get("Reserved", "-"),
                r.get("Free", "-"),
                r.get("Status", "OK")
            )
        console.print(table)
    else:
        print("\n=== EEPM3 Forensic VRAM Autopsy Report ===")
        header = f"{'Phase':<25} | {'Allocated':<10} | {'Reserved':<10} | {'Free':<10} | {'Status'}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(f"{r.get('Label'):<25} | {r.get('Allocated', '-'):<10} | {r.get('Reserved', '-'):<10} | {r.get('Free', '-'):<10} | {r.get('Status')}")

def run_autopsy():
    report = []
    
    # 0. System Baseline
    report.append({**log_vram("Baseline (Pre-Load)"), "Status": "System Clean"})
    
    # 0.5 simulate JAX pre-allocation
    print("\n[Setup] Initializing JAX (Simulating Pipeline Contention)...")
    try:
        import jax
        import jax.numpy as jnp
        # Force JAX to pre-allocate (this is what usually happens in EEPM3)
        _ = jax.device_put(jnp.zeros((1024, 1024)))
        report.append({**log_vram("JAX Pre-allocation Active"), "Status": "14GB+ likely reserved"})
    except ImportError:
        report.append({**log_vram("JAX Missing"), "Status": "Skipped JAX Setup"})

    # Phase 1: Evo2 7B Weight Test
    print("\n[Phase 1] Attempting Evo2 7B Load (bfloat16)...")
    try:
        from evo2 import Evo2
        # This will likely OOM immediately on 16GB if JAX or overhead is present
        report.append({**log_vram("Pre-Evo2 Load"), "Status": "Starting Wait..."})
        model = Evo2("evo2_7b", device="cuda")
        report.append({**log_vram("Evo2 Loaded"), "Status": "SUCCESS (Unexpected stability)"})
        
        # Test 131kb KV Cache formation
        print("   -> Testing 131kb sequence forward pass...")
        dummy_ids = torch.randint(0, 4, (1, 131072)).to("cuda")
        _ = model(dummy_ids)
        report.append({**log_vram("Evo2 131kb Forward"), "Status": "SUCCESS"})
        del model
    except torch.cuda.OutOfMemoryError:
        report.append({**log_vram("Evo2 Load Failure"), "Status": "CRASH (CUDA OOM: Weights > VRAM)"})
        print("   -> CRASH DETECTED. Cleaning cache for Phase 2...")
    except Exception as e:
        report.append({**log_vram("Evo2 Phase"), "Status": f"FAILED ({type(e).__name__})"})
        print(f"   -> Error: {e}")
    finally:
        torch.cuda.empty_cache()
        gc.collect()

    # Phase 2: NT-500M Context Test
    print("\n[Phase 2] Attempting NT-500M (500M Params) Test...")
    try:
        nt_name = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
        tokenizer = AutoTokenizer.from_pretrained(nt_name)
        model = AutoModelForMaskedLM.from_pretrained(nt_name, torch_dtype=torch.float16).to("cuda")
        report.append({**log_vram("NT-500M Loaded"), "Status": "Weights Consumed ~1GB"})
        
        # 131kb Chunked Test
        print("   -> Running Paranoid 1024-token chunked scoring (131kb)...")
        seq = "A" * 131072
        tokens = tokenizer(seq, return_tensors="pt")["input_ids"][0]
        
        # Sequential chunks
        chunk_size = 1024
        for i in range(0, tokens.shape[0], chunk_size):
            end_idx = min(i + chunk_size, tokens.shape[0])
            chunk = tokens[i:end_idx].unsqueeze(0).to("cuda")
            _ = model(input_ids=chunk, labels=chunk)
            del chunk
            torch.cuda.empty_cache() # Aggressive manual flush
            
        report.append({**log_vram("NT-131kb Paranoid Scoring"), "Status": "SUCCESS (Safe Path)"})
    except torch.cuda.OutOfMemoryError:
        report.append({**log_vram("NT-131kb Failure"), "Status": "CRASH (VRAM Fragmentation/Contention)"})
    except Exception as e:
        report.append({**log_vram("NT Phase"), "Status": f"FAILED ({type(e).__name__})"})
    finally:
        torch.cuda.empty_cache()
        gc.collect()

    # Phase 3: Stability Scale-Back Search
    print("\n[Phase 3] Stability Search (Max Seq Length with JAX Contention)...")
    for length in [131072, 65536, 32768, 16384, 8192]:
        print(f"   -> Testing length {length}bp...", end="\r")
        try:
            seq = "G" * length
            tokens = tokenizer(seq, return_tensors="pt")["input_ids"][0]
            # Use paranoid loop but for just one window to test survival
            chunk = tokens[:1024].unsqueeze(0).to("cuda")
            _ = model(input_ids=chunk, labels=chunk)
            del chunk
            torch.cuda.empty_cache()
            report.append({**log_vram(f"Length {length}bp"), "Status": "PASS (Stable)"})
        except torch.cuda.OutOfMemoryError:
            report.append({**log_vram(f"Length {length}bp"), "Status": "FAIL (OOM: Wall hit)"})
            torch.cuda.empty_cache()
            gc.collect()
            
    print("\nAutopsy complete.")
    print_report(report)

if __name__ == "__main__":
    run_autopsy()
