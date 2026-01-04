from pathlib import Path
from collections import Counter

mp_data = Path("MP_DATA_QUALITY")
counts = {d.name: len(list(d.glob("*.npy"))) for d in mp_data.iterdir() if d.is_dir()}

print("Sample distribution in MP_DATA_QUALITY:")
print("="*60)
for sign, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{sign:40s}: {count:3d} samples")

print("\n" + "="*60)
print(f"Total classes: {len(counts)}")
print(f"Total samples: {sum(counts.values())}")
print(f"Average: {sum(counts.values())/len(counts):.1f} samples/class")
print(f"Min: {min(counts.values())}, Max: {max(counts.values())}")
