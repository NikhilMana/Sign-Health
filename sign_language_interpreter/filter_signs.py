"""
Filter MP_DATA to keep only signs with enough samples
"""
from pathlib import Path
from collections import Counter
import shutil

def analyze_dataset():
    mp_data = Path("MP_DATA")
    
    if not mp_data.exists():
        print("MP_DATA folder not found!")
        return
    
    sign_counts = {}
    for sign_folder in mp_data.iterdir():
        if sign_folder.is_dir():
            count = len(list(sign_folder.glob("*.npy")))
            sign_counts[sign_folder.name] = count
    
    print(f"\nTotal signs: {len(sign_counts)}")
    print(f"Total samples: {sum(sign_counts.values())}")
    
    print("\n" + "="*60)
    print("SIGNS WITH 10+ SAMPLES (will be kept):")
    print("="*60)
    kept = {k: v for k, v in sorted(sign_counts.items(), key=lambda x: x[1], reverse=True) if v >= 10}
    for sign, count in kept.items():
        print(f"{sign}: {count} samples")
    
    print("\n" + "="*60)
    print(f"SIGNS WITH <10 SAMPLES (will be filtered out): {len(sign_counts) - len(kept)}")
    print("="*60)
    removed = {k: v for k, v in sorted(sign_counts.items(), key=lambda x: x[1], reverse=True) if v < 10}
    for sign, count in list(removed.items())[:20]:
        print(f"{sign}: {count} samples")
    if len(removed) > 20:
        print(f"... and {len(removed) - 20} more")
    
    print(f"\n\nSUMMARY:")
    print(f"  Signs to keep: {len(kept)}")
    print(f"  Signs to remove: {len(removed)}")
    print(f"  Samples to keep: {sum(kept.values())}")
    print(f"  Samples to remove: {sum(removed.values())}")

if __name__ == "__main__":
    analyze_dataset()
