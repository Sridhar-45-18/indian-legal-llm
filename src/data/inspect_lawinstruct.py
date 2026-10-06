from datasets import load_from_disk
from pathlib import Path
import json


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(
    r"C:\ALL_PROJECTS\PREDECTIVE"
)

DATASET_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "lawinstruct"
)

OUTPUT_DIR = (
    DATASET_DIR
    / "inspection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Load dataset
# ============================================================

print("=" * 70)
print("LawInstruct Dataset Inspection")
print("=" * 70)

print("\nDataset location:")
print(DATASET_DIR)

try:

    dataset = load_from_disk(
        str(DATASET_DIR)
    )

except Exception as e:

    print("\nERROR loading dataset:")
    print(e)

    raise SystemExit


# ============================================================
# Dataset information
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(f"\nDataset type: {type(dataset)}")

if hasattr(dataset, "num_rows"):
    print(f"Number of rows: {dataset.num_rows:,}")

if hasattr(dataset, "column_names"):
    print(
        f"Columns: {dataset.column_names}"
    )


# ============================================================
# Inspect dataset splits
# ============================================================

if hasattr(dataset, "keys"):

    print("\nDataset splits:")

    for split_name in dataset.keys():

        split = dataset[split_name]

        print(
            f"  {split_name}: "
            f"{split.num_rows:,} rows"
        )


# ============================================================
# Determine the dataset to inspect
# ============================================================

if hasattr(dataset, "keys"):

    split_names = list(dataset.keys())

    if not split_names:

        print("\nNo dataset splits found.")
        raise SystemExit

    data = dataset[split_names[0]]

else:

    data = dataset


# ============================================================
# Inspect columns
# ============================================================

columns = data.column_names

print("\n" + "=" * 70)
print("COLUMNS")
print("=" * 70)

for column in columns:

    print(f"  {column}")


# ============================================================
# Inspect first 5 examples
# ============================================================

print("\n" + "=" * 70)
print("FIRST 5 EXAMPLES")
print("=" * 70)

sample_count = min(
    5,
    len(data)
)

sample_records = []

for i in range(sample_count):

    example = data[i]

    sample_records.append(example)

    print(f"\n--- Example {i + 1} ---")

    for key, value in example.items():

        print(f"\n{key}:")

        if isinstance(value, str):

            if len(value) > 1500:

                print(
                    value[:1500]
                    + "\n..."
                )

            else:

                print(value)

        else:

            print(value)


# ============================================================
# Save sample
# ============================================================

sample_file = (
    OUTPUT_DIR
    / "sample_5.json"
)

with open(
    sample_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        sample_records,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# Text statistics
# ============================================================

print("\n" + "=" * 70)
print("TEXT FIELD STATISTICS")
print("=" * 70)

for column in columns:

    values = []

    for i in range(
        min(len(data), 5000)
    ):

        value = data[i][column]

        if isinstance(value, str):

            value = value.strip()

            if value:

                values.append(
                    len(value)
                )

    if values:

        print(f"\n{column}")

        print(
            f"  Non-empty: {len(values):,}"
        )

        print(
            f"  Minimum:   {min(values):,} chars"
        )

        print(
            f"  Maximum:   {max(values):,} chars"
        )

        print(
            f"  Average:   "
            f"{sum(values) / len(values):,.2f} chars"
        )


# ============================================================
# Search for India / Hindi relevance
# ============================================================

print("\n" + "=" * 70)
print("INDIA / HINDI RELEVANCE CHECK")
print("=" * 70)

india_keywords = [
    "india",
    "indian",
    "bharat",
    "supreme court of india",
    "high court",
    "delhi",
    "mumbai",
    "bihar",
    "maharashtra",
    "uttar pradesh",
    "karnataka",
    "tamil nadu",
    "hindi"
]

keyword_counts = {
    keyword: 0
    for keyword in india_keywords
}

rows_checked = min(
    len(data),
    5000
)

for i in range(rows_checked):

    example = data[i]

    combined_text = " ".join(
        str(value)
        for value in example.values()
        if value is not None
    ).lower()

    for keyword in india_keywords:

        if keyword in combined_text:

            keyword_counts[keyword] += 1


print(
    f"\nRows checked: {rows_checked:,}"
)

for keyword, count in keyword_counts.items():

    print(
        f"{keyword:<25} "
        f"{count:>5}"
    )


# ============================================================
# Save inspection summary
# ============================================================

summary = {
    "dataset_location": str(DATASET_DIR),
    "columns": columns,
    "rows_checked_for_statistics": rows_checked,
    "india_hindi_keyword_counts": keyword_counts
}

summary_file = (
    OUTPUT_DIR
    / "inspection_summary.json"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# Final output
# ============================================================

print("\n" + "=" * 70)
print("LAWInstruct INSPECTION COMPLETE")
print("=" * 70)

print("\nSample saved to:")
print(sample_file)

print("\nSummary saved to:")
print(summary_file)

print(
    "\nDo NOT preprocess or select the 15K-20K examples yet."
)

print(
    "First inspect the output and determine the actual "
    "LawInstruct schema and India/Hindi filtering strategy."
)