from datasets import load_from_disk
from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALPACA_PATH = PROJECT_ROOT / "data" / "raw" / "alpaca"


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("ALPACA DATASET INSPECTION")
print("=" * 70)

print(f"\nDataset path:\n{ALPACA_PATH}")

dataset = load_from_disk(str(ALPACA_PATH))

print("\nDataset loaded successfully!")
print(dataset)


# ============================================================
# BASIC INFORMATION
# ============================================================

train = dataset

print("\n" + "=" * 70)
print("BASIC INFORMATION")
print("=" * 70)

print(f"Number of rows : {len(train)}")
print(f"Columns        : {train.column_names}")


# ============================================================
# FEATURES
# ============================================================

print("\n" + "=" * 70)
print("DATASET FEATURES")
print("=" * 70)

print(train.features)


# ============================================================
# CONVERT TO PANDAS
# ============================================================

df = train.to_pandas()

print("\nPandas shape:", df.shape)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(df.isnull().sum())


# ============================================================
# DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATES")
print("=" * 70)

print("Exact duplicate rows:", df.duplicated().sum())

if "instruction" in df.columns:
    print(
        "Duplicate instructions:",
        df["instruction"].duplicated().sum()
    )


# ============================================================
# TEXT LENGTH ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("TEXT LENGTH ANALYSIS")
print("=" * 70)

for column in ["instruction", "input", "output", "text"]:

    if column in df.columns:

        lengths = df[column].fillna("").astype(str).str.len()

        print(f"\n{column}")
        print(f"  Minimum : {lengths.min()}")
        print(f"  Maximum : {lengths.max()}")
        print(f"  Mean    : {lengths.mean():.2f}")
        print(f"  Median  : {lengths.median():.2f}")


# ============================================================
# EMPTY VALUES
# ============================================================

print("\n" + "=" * 70)
print("EMPTY TEXT VALUES")
print("=" * 70)

for column in ["instruction", "input", "output", "text"]:

    if column in df.columns:

        empty_count = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        print(f"{column}: {empty_count}")


# ============================================================
# SAMPLE RECORDS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE RECORDS")
print("=" * 70)

for i in range(min(5, len(df))):

    print(f"\n--- SAMPLE {i + 1} ---")

    for column in df.columns:
        print(f"\n{column}:")
        print(df.iloc[i][column])


# ============================================================
# LONGEST OUTPUTS
# ============================================================

if "output" in df.columns:

    print("\n" + "=" * 70)
    print("5 LONGEST OUTPUTS")
    print("=" * 70)

    output_lengths = (
        df["output"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    longest_indices = output_lengths.nlargest(5).index

    for i, idx in enumerate(longest_indices, 1):

        print(f"\n--- LONGEST OUTPUT {i} ---")
        print("Length:", output_lengths[idx])
        print(df.loc[idx, "output"])


# ============================================================
# SHORT OUTPUTS
# ============================================================

if "output" in df.columns:

    print("\n" + "=" * 70)
    print("OUTPUTS WITH LESS THAN 20 CHARACTERS")
    print("=" * 70)

    output_lengths = (
        df["output"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    short_count = (output_lengths < 20).sum()

    print("Count:", short_count)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

print(f"""
Rows              : {len(df)}
Columns           : {list(df.columns)}
Duplicate rows    : {df.duplicated().sum()}
""")