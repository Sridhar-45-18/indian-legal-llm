from datasets import load_dataset, concatenate_datasets
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bns_bnss_bsa"

DATASET_NAME = "GSMS-B/Indian-Legal-QA-BNS-BNSS-BSA"

print("=" * 70)
print("DOWNLOADING BNS-BNSS-BSA LEGAL QA DATASET")
print("=" * 70)

print(f"\nDataset      : {DATASET_NAME}")
print(f"Download to  : {RAW_DIR}")

RAW_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Load BNS
# ------------------------------------------------------------

print("\n[1/3] Loading BNS QA dataset...")

bns = load_dataset(
    DATASET_NAME,
    data_files="bns_legal_qa.jsonl",
    split="train"
)

print(f"BNS rows: {len(bns)}")
print(f"BNS columns: {bns.column_names}")

# ------------------------------------------------------------
# Load BNSS
# ------------------------------------------------------------

print("\n[2/3] Loading BNSS QA dataset...")

bnss = load_dataset(
    DATASET_NAME,
    data_files="bnss_legal_qa.jsonl",
    split="train"
)

print(f"BNSS rows: {len(bnss)}")
print(f"BNSS columns: {bnss.column_names}")

# ------------------------------------------------------------
# Load BSA
# ------------------------------------------------------------

print("\n[3/3] Loading BSA QA dataset...")

bsa = load_dataset(
    DATASET_NAME,
    data_files="bsa_legal_qa.jsonl",
    split="train"
)

print(f"BSA rows: {len(bsa)}")
print(f"BSA columns: {bsa.column_names}")

# ------------------------------------------------------------
# Combine
# ------------------------------------------------------------

print("\nCombining BNS + BNSS + BSA...")

dataset = concatenate_datasets([bns, bnss, bsa])

print("\nCombined dataset:")
print(dataset)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

print("\nSaving dataset to disk...")

dataset.save_to_disk(str(RAW_DIR))

print("\n" + "=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

print(f"Rows       : {len(dataset)}")
print(f"Columns    : {dataset.column_names}")
print(f"Saved at   : {RAW_DIR}")

print("\nExpected rows: 6354")

if len(dataset) == 6354:
    print("✓ Row count verified successfully.")
else:
    print("WARNING: Row count differs from expected 6354.")