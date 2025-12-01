#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data loading test script for GNNtest2
Tests whether CFD data can be loaded correctly.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Use the original data loader functions
from utils.data_loader import find_time_list, load_case_with_csr

def test_original_format():
    """
    Test with original data format (rank-based files).

    Expected file structure:
    data_dir/
        ├── pEqn_0.001_rank0.dat
        ├── x_0.001_rank0.dat
        └── A_csr_0.001.dat
    """
    print("="*60)
    print("Testing Original Data Format")
    print("="*60)

    # Adjust these paths to your actual data location
    data_dir = "../cylinder/work/data/gnn"  # Path to your gnn directory
    rank_str = "0"  # Your rank string (check your file names)

    print(f"Data directory: {data_dir}")
    print(f"Rank string: {rank_str}")
    print()

    # Check if directory exists
    if not Path(data_dir).exists():
        print(f"❌ Directory not found: {data_dir}")
        print("Please update data_dir to point to your gnn directory")
        return False

    # List files in directory
    print("Files in directory:")
    gnn_path = Path(data_dir)
    files = sorted(gnn_path.glob("*"))
    for f in files[:10]:  # Show first 10 files
        print(f"  {f.name}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more files")
    print()

    # Find available time steps
    try:
        time_list = find_time_list(data_dir, rank_str)
        print(f"✓ Found {len(time_list)} time steps")
        if len(time_list) > 0:
            print(f"  Time steps: {time_list[:5]}")
            if len(time_list) > 5:
                print(f"  ... and {len(time_list) - 5} more")
        else:
            print("❌ No time steps found!")
            print(f"Expected files like: pEqn_*_rank{rank_str}.dat")
            return False
    except Exception as e:
        print(f"❌ Error finding time steps: {e}")
        return False

    # Try loading first time step
    if len(time_list) > 0:
        t = time_list[0]
        print(f"\nLoading time step: {t}")
        try:
            data = load_case_with_csr(data_dir, t, rank_str)

            print(f"✓ Data loaded successfully!")
            print(f"  Features shape: {data['feats_np'].shape}")
            print(f"  Solution shape: {data['x_true_np'].shape}")
            print(f"  Edge index shape: {data['edge_index_np'].shape}")
            print(f"  CSR matrix nnz: {len(data['vals_np'])}")
            print(f"  Number of cells: {data['feats_np'].shape[0]}")

            # Show feature ranges
            feats = data['feats_np']
            print(f"\n  Feature ranges:")
            print(f"    Coordinates (x,y,z): [{feats[:,:3].min():.3f}, {feats[:,:3].max():.3f}]")
            print(f"    Skewness: [{feats[:,5].min():.3f}, {feats[:,5].max():.3f}]")
            print(f"    Non-orthogonality: [{feats[:,6].min():.3f}, {feats[:,6].max():.3f}]")
            print(f"    Aspect ratio: [{feats[:,7].min():.3f}, {feats[:,7].max():.3f}]")

            return True

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False

    return True


def find_correct_rank():
    """
    Auto-detect the correct rank string from file names.
    """
    data_dir = "../cylinder/work/data/gnn"

    if not Path(data_dir).exists():
        print(f"Directory not found: {data_dir}")
        return None

    # Look for pEqn files
    gnn_path = Path(data_dir)
    pEqn_files = list(gnn_path.glob("pEqn_*_rank*.dat"))

    if len(pEqn_files) == 0:
        print("No pEqn_*_rank*.dat files found")
        return None

    # Extract rank from first file
    # Example: pEqn_0.001_rank0.dat -> rank0
    fname = pEqn_files[0].name
    parts = fname.split('_rank')
    if len(parts) == 2:
        rank_part = parts[1].replace('.dat', '')
        print(f"Auto-detected rank: {rank_part}")
        return rank_part

    return None


if __name__ == '__main__':
    print("GNNtest2 Data Loading Test")
    print()

    # First, try to auto-detect rank
    print("Attempting to auto-detect data format...")
    rank = find_correct_rank()

    if rank is not None:
        print(f"\nUsing rank string: {rank}")
        print("If this is incorrect, please modify the test script.\n")
    else:
        print("\nCould not auto-detect rank. Using default: 0\n")
        rank = "0"

    # Run test
    success = test_original_format()

    if success:
        print("\n" + "="*60)
        print("✓ DATA LOADING TEST PASSED!")
        print("="*60)
        print("\nYou can now proceed to run the training scripts.")
        print("\nNext step:")
        print("  cd experiments")
        print("  python train_baseline.py --data_dir ../../cylinder/work/data/gnn --epochs 10")
    else:
        print("\n" + "="*60)
        print("❌ DATA LOADING TEST FAILED")
        print("="*60)
        print("\nPlease check:")
        print("  1. Data directory path is correct")
        print("  2. Files exist in expected format (pEqn_*_rank*.dat)")
        print("  3. Rank string matches your file names")
