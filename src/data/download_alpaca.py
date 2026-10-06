from datasets import load_dataset
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw dataset location
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "alpaca"

print("=" * 70)
print("ALPACA DATASET DOWNLOAD")
print("=" * 70)

print(f"Project root : {PROJECT_ROOT}")
print(f"Output path  : {OUTPUT_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nDownloading Alpaca...")
print("Dataset: tatsu-lab/alpaca")

dataset = load_dataset(
    "tatsu-lab/alpaca",
    split="train"
)

print("\nDataset downloaded successfully!")
print(dataset)

print("\nNumber of examples:", len(dataset))
print("Columns:", dataset.column_names)

print("\nSaving dataset locally...")
dataset.save_to_disk(str(OUTPUT_DIR))

print("\n" + "=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)
print(f"Saved to: {OUTPUT_DIR}")