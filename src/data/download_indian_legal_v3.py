from datasets import load_dataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "indian_legal_v3"

print("=" * 70)
print("DOWNLOADING INDIAN LEGAL DATASET V3")
print("=" * 70)

print(f"\nDataset      : kaushik-harsh-99/Indian-legal-data-v3")
print(f"Download to  : {RAW_DIR}")

RAW_DIR.mkdir(parents=True, exist_ok=True)

print("\nDownloading dataset...")
print("This dataset is approximately 638 MB, so it may take some time.")

dataset = load_dataset(
    "kaushik-harsh-99/Indian-legal-data-v3",
    "context_less_than_4096",
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