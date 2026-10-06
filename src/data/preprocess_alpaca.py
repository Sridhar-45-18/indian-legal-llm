from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from pathlib import Path
import re
import json


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MAX_SEQ_LENGTH = 2048
MIN_OUTPUT_TOKENS = 20
VALIDATION_RATIO = 0.10
SEED = 42


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "alpaca"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "stage1_general"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text):
    """Clean and normalize text while preserving meaning."""

    if text is None:
        return ""

    text = str(text)

    # Remove null/control characters
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_training_text(instruction, input_text, output):
    """Create the text used for token-length checking."""

    if input_text:
        return (
            f"Instruction:\n{instruction}\n\n"
            f"Input:\n{input_text}\n\n"
            f"Response:\n{output}"
        )

    return (
        f"Instruction:\n{instruction}\n\n"
        f"Response:\n{output}"
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("ALPACA PREPROCESSING")
print("=" * 70)

print(f"\nProject root : {PROJECT_ROOT}")
print(f"Raw dataset  : {RAW_DIR}")
print(f"Output       : {OUTPUT_DIR}")

print("\nLoading raw Alpaca dataset...")

dataset = load_from_disk(str(RAW_DIR))

print(dataset)

# load_from_disk returns Dataset for our Alpaca dataset
df = dataset.to_pandas()

print(f"\nOriginal number of rows: {len(df)}")


# ============================================================
# STEP 1 — SELECT REQUIRED COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 — SELECTING COLUMNS")
print("=" * 70)

df = df[["instruction", "input", "output"]].copy()

print("Columns:", list(df.columns))


# ============================================================
# STEP 2 — NORMALIZE TEXT
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 — NORMALIZING TEXT")
print("=" * 70)

for column in ["instruction", "input", "output"]:
    df[column] = df[column].apply(clean_text)

print("Text normalization complete.")


# ============================================================
# STEP 3 — REMOVE EMPTY OUTPUTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 — REMOVING EMPTY OUTPUTS")
print("=" * 70)

before = len(df)

df = df[df["output"].str.len() > 0].copy()

removed = before - len(df)

print(f"Removed empty outputs: {removed}")
print(f"Remaining rows: {len(df)}")


# ============================================================
# STEP 4 — REMOVE DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 — REMOVING DUPLICATES")
print("=" * 70)

before = len(df)

df = df.drop_duplicates(
    subset=["instruction", "input", "output"]
).copy()

removed = before - len(df)

print(f"Removed duplicate records: {removed}")
print(f"Remaining rows: {len(df)}")


# ============================================================
# STEP 5 — LOAD QWEN TOKENIZER
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 — LOADING QWEN2.5-1.5B TOKENIZER")
print("=" * 70)

print(f"Tokenizer: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded successfully.")


# ============================================================
# STEP 6 — TOKEN LENGTH FILTERING
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 — TOKEN LENGTH FILTERING")
print("=" * 70)

print(f"Minimum output tokens : {MIN_OUTPUT_TOKENS}")
print(f"Maximum total tokens  : {MAX_SEQ_LENGTH}")

output_token_counts = []
total_token_counts = []

valid_rows = []

removed_short_output = 0
removed_long_sequence = 0

for index, row in df.iterrows():

    instruction = row["instruction"]
    input_text = row["input"]
    output = row["output"]

    # Tokenize output
    output_tokens = tokenizer.encode(
        output,
        add_special_tokens=False
    )

    output_length = len(output_tokens)

    # Build complete example for total length calculation
    full_text = build_training_text(
        instruction,
        input_text,
        output
    )

    full_tokens = tokenizer.encode(
        full_text,
        add_special_tokens=True
    )

    total_length = len(full_tokens)

    output_token_counts.append(output_length)
    total_token_counts.append(total_length)

    # Filter short outputs
    if output_length < MIN_OUTPUT_TOKENS:
        removed_short_output += 1
        continue

    # Filter sequences longer than 2048 tokens
    if total_length > MAX_SEQ_LENGTH:
        removed_long_sequence += 1
        continue

    valid_rows.append(index)


# Keep only valid examples
df = df.loc[valid_rows].copy()

print(f"\nRemoved short outputs (<20 tokens): {removed_short_output}")
print(f"Removed long sequences (>2048 tokens): {removed_long_sequence}")
print(f"Remaining rows: {len(df)}")


# ============================================================
# TOKEN STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("TOKEN STATISTICS")
print("=" * 70)

if output_token_counts:
    print(
        f"Original output token count - "
        f"min: {min(output_token_counts)}, "
        f"max: {max(output_token_counts)}, "
        f"average: {sum(output_token_counts) / len(output_token_counts):.2f}"
    )

if total_token_counts:
    print(
        f"Original total token count - "
        f"min: {min(total_token_counts)}, "
        f"max: {max(total_token_counts)}, "
        f"average: {sum(total_token_counts) / len(total_token_counts):.2f}"
    )


# ============================================================
# STEP 7 — RESET INDEX
# ============================================================

df = df.reset_index(drop=True)


# ============================================================
# STEP 8 — CREATE DATASET
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 — CREATING PROCESSED DATASET")
print("=" * 70)

processed_dataset = Dataset.from_pandas(
    df,
    preserve_index=False
)

print(processed_dataset)


# ============================================================
# STEP 9 — TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 — TRAIN / VALIDATION SPLIT")
print("=" * 70)

split_dataset = processed_dataset.train_test_split(
    test_size=VALIDATION_RATIO,
    seed=SEED
)

train_dataset = split_dataset["train"]
validation_dataset = split_dataset["test"]

print(f"Training examples   : {len(train_dataset)}")
print(f"Validation examples : {len(validation_dataset)}")


# ============================================================
# STEP 10 — SAVE PARQUET
# ============================================================

print("\n" + "=" * 70)
print("STEP 10 — SAVING PARQUET FILES")
print("=" * 70)

train_parquet = OUTPUT_DIR / "train.parquet"
validation_parquet = OUTPUT_DIR / "validation.parquet"

train_dataset.to_parquet(str(train_parquet))
validation_dataset.to_parquet(str(validation_parquet))

print(f"Saved: {train_parquet}")
print(f"Saved: {validation_parquet}")


# ============================================================
# STEP 11 — SAVE JSONL
# ============================================================

print("\n" + "=" * 70)
print("STEP 11 — SAVING JSONL FILES")
print("=" * 70)


def save_jsonl(dataset, path):

    with open(path, "w", encoding="utf-8") as f:

        for example in dataset:

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                ) + "\n"
            )


train_jsonl = OUTPUT_DIR / "train.jsonl"
validation_jsonl = OUTPUT_DIR / "validation.jsonl"

save_jsonl(train_dataset, train_jsonl)
save_jsonl(validation_dataset, validation_jsonl)

print(f"Saved: {train_jsonl}")
print(f"Saved: {validation_jsonl}")


# ============================================================
# STEP 12 — SAVE SUMMARY
# ============================================================

summary = {
    "source_dataset": "tatsu-lab/alpaca",
    "original_rows": 52002,
    "processed_rows": len(processed_dataset),
    "train_rows": len(train_dataset),
    "validation_rows": len(validation_dataset),
    "validation_ratio": VALIDATION_RATIO,
    "seed": SEED,
    "tokenizer": MODEL_NAME,
    "max_sequence_tokens": MAX_SEQ_LENGTH,
    "minimum_output_tokens": MIN_OUTPUT_TOKENS,
    "removed_short_outputs": removed_short_output,
    "removed_long_sequences": removed_long_sequence,
}

summary_path = OUTPUT_DIR / "preprocessing_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("ALPACA PREPROCESSING COMPLETE")
print("=" * 70)

print("\nFinal output:")

print(f"  Train       : {train_parquet}")
print(f"  Validation  : {validation_parquet}")
print(f"  Train JSONL : {train_jsonl}")
print(f"  Valid JSONL : {validation_jsonl}")
print(f"  Summary     : {summary_path}")

print("\nFinal dataset sizes:")
print(f"  Train       : {len(train_dataset)}")
print(f"  Validation  : {len(validation_dataset)}")
print(f"  Total       : {len(train_dataset) + len(validation_dataset)}")

print("\nStage 1 preprocessing finished successfully.")