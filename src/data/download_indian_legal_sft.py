from datasets import load_dataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "indian_legal_sft"

print("=" * 70)
print("DOWNLOADING INDIAN LEGAL SFT DATASET")
print("=" * 70)

print(f"\nDataset      : SharathReddy/Indian-Legal-SFT-Dataset")
print(f"Download to  : {RAW_DIR}")

RAW_DIR.mkdir(parents=True, exist_ok=True)

print("\nDownloading dataset...")

dataset = load_dataset(
    "SharathReddy/Indian-Legal-SFT-Dataset",
    split="train"
)

print("\nDataset downloaded successfully!")
print(dataset)

print("\nSaving to disk...")
dataset.save_to_disk(str(RAW_DIR))

print("\n" + "=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

print(f"Rows       : {len(dataset)}")
print(f"Columns    : {dataset.column_names}")
print(f"Saved at   : {RAW_DIR}")