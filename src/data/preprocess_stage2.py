from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
from pathlib import Path
import json
import re


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "stage2_legal"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B"

MIN_OUTPUT_TOKENS = 20
MAX_TOTAL_TOKENS = 2048

RANDOM_SEED = 42


print("=" * 70)
print("STAGE 2 — LEGAL DATA PREPROCESSING")
print("=" * 70)

print(f"\nProject root : {PROJECT_ROOT}")
print(f"Output       : {OUTPUT_DIR}")


# ============================================================
# STEP 1 — LOAD DATASETS
# ============================================================

print("\n" + "=" * 70)
print("STEP 1 — LOADING STAGE 2 DATASETS")
print("=" * 70)


print("\n[1/4] Loading Indian Legal V3...")

v3 = load_from_disk(
    str(RAW_DIR / "indian_legal_v3")
)

print(f"Rows    : {len(v3)}")
print(f"Columns : {v3.column_names}")


print("\n[2/4] Loading Indian Legal SFT...")

legal_sft = load_from_disk(
    str(RAW_DIR / "indian_legal_sft")
)

print(f"Rows    : {len(legal_sft)}")
print(f"Columns : {legal_sft.column_names}")


print("\n[3/4] Loading BNS-BNSS-BSA...")

bns = load_from_disk(
    str(RAW_DIR / "bns_bnss_bsa")
)

print(f"Rows    : {len(bns)}")
print(f"Columns : {bns.column_names}")


print("\n[4/4] Loading Constitution Instruction Set...")

constitution = load_from_disk(
    str(RAW_DIR / "constitution")
)

print(f"Rows    : {len(constitution)}")
print(f"Columns : {constitution.column_names}")


# ============================================================
# STEP 2 — NORMALIZE DATASETS
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 — NORMALIZING DATASETS")
print("=" * 70)


def clean_text(value):
    """Normalize whitespace and convert values to strings."""

    if value is None:
        return ""

    value = str(value)

    value = value.replace("\r\n", "\n")
    value = value.replace("\r", "\n")

    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)

    return value.strip()


# ------------------------------------------------------------
# Indian Legal V3
# ------------------------------------------------------------

print("\nNormalizing Indian Legal V3...")


def normalize_v3(example):

    return {
        "instruction": clean_text(example["instruction"]),
        "input": "",
        "output": clean_text(example["output"])
    }


v3 = v3.map(normalize_v3)

v3 = v3.select_columns(
    ["instruction", "input", "output"]
)

print(f"Normalized V3 rows: {len(v3)}")


# ------------------------------------------------------------
# Indian Legal SFT
# ------------------------------------------------------------

print("\nNormalizing Indian Legal SFT...")


def normalize_legal_sft(example):

    return {
        "instruction": clean_text(example["instruction"]),
        "input": clean_text(example["context"]),
        "output": clean_text(example["response"])
    }


legal_sft = legal_sft.map(normalize_legal_sft)

legal_sft = legal_sft.select_columns(
    ["instruction", "input", "output"]
)

print(f"Normalized Legal SFT rows: {len(legal_sft)}")


# ------------------------------------------------------------
# BNS-BNSS-BSA
# ------------------------------------------------------------

print("\nNormalizing BNS-BNSS-BSA...")


def normalize_bns(example):

    act = clean_text(example["act"])
    section_number = clean_text(example["section_number"])
    section_title = clean_text(example["section_title"])

    question = clean_text(example["question"])
    answer = clean_text(example["answer"])

    context_parts = []

    if act:
        context_parts.append(f"Act: {act}")

    if section_number:
        context_parts.append(
            f"Section: {section_number}"
        )

    if section_title:
        context_parts.append(
            f"Section Title: {section_title}"
        )

    context = "\n".join(context_parts)

    return {
        "instruction": question,
        "input": context,
        "output": answer
    }


bns = bns.map(normalize_bns)

bns = bns.select_columns(
    ["instruction", "input", "output"]
)

print(f"Normalized BNS-BNSS-BSA rows: {len(bns)}")


# ------------------------------------------------------------
# Constitution
# ------------------------------------------------------------

print("\nNormalizing Constitution dataset...")


def normalize_constitution(example):

    return {
        "instruction": clean_text(example["instruction"]),
        "input": clean_text(example["input"]),
        "output": clean_text(example["output"])
    }


constitution = constitution.map(normalize_constitution)

constitution = constitution.select_columns(
    ["instruction", "input", "output"]
)

print(f"Normalized Constitution rows: {len(constitution)}")


# ============================================================
# STEP 3 — COMBINE DATASETS
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 — COMBINING STAGE 2 DATASETS")
print("=" * 70)


combined = concatenate_datasets(
    [
        v3,
        legal_sft,
        bns,
        constitution
    ]
)

print("\nCombined dataset:")
print(combined)

print(
    f"\nTotal rows before filtering: {len(combined)}"
)


# ============================================================
# STEP 4 — REMOVE EMPTY OUTPUTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 — REMOVING EMPTY OUTPUTS")
print("=" * 70)


before = len(combined)

combined = combined.filter(
    lambda x: len(x["output"].strip()) > 0
)

removed_empty = before - len(combined)

print(
    f"Removed empty outputs : {removed_empty}"
)

print(
    f"Remaining rows        : {len(combined)}"
)


# ============================================================
# STEP 5 — REMOVE DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 — REMOVING DUPLICATES")
print("=" * 70)


before = len(combined)

print("\nCreating duplicate keys...")

combined = combined.map(
    lambda x: {
        "duplicate_key": (
            x["instruction"]
            + "\n"
            + x["input"]
            + "\n"
            + x["output"]
        )
    }
)

seen = set()
keep_indices = []

for i, key in enumerate(
    combined["duplicate_key"]
):

    if key not in seen:

        seen.add(key)
        keep_indices.append(i)


combined = combined.select(
    keep_indices
)

combined = combined.remove_columns(
    ["duplicate_key"]
)

removed_duplicates = before - len(combined)

print(
    f"Removed duplicates : {removed_duplicates}"
)

print(
    f"Remaining rows     : {len(combined)}"
)


# ============================================================
# STEP 6 — LOAD QWEN TOKENIZER
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 — LOADING QWEN2.5-1.5B TOKENIZER")
print("=" * 70)


print(
    f"Tokenizer: {TOKENIZER_NAME}"
)

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME
)

print(
    "Tokenizer loaded successfully."
)


# ============================================================
# STEP 7 — TOKEN LENGTH FILTERING
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 — TOKEN LENGTH FILTERING")
print("=" * 70)


print(
    f"Minimum output tokens : {MIN_OUTPUT_TOKENS}"
)

print(
    f"Maximum total tokens  : {MAX_TOTAL_TOKENS}"
)


def get_token_counts(example):

    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]

    if input_text.strip():

        prompt_text = (
            "Instruction:\n"
            + instruction
            + "\n\n"
            + "Input:\n"
            + input_text
            + "\n\n"
            + "Response:\n"
        )

    else:

        prompt_text = (
            "Instruction:\n"
            + instruction
            + "\n\n"
            + "Response:\n"
        )

    output_tokens = tokenizer(
        output,
        add_special_tokens=False
    )["input_ids"]

    total_tokens = tokenizer(
        prompt_text + output,
        add_special_tokens=True
    )["input_ids"]

    return {
        "output_token_count": len(output_tokens),
        "total_token_count": len(total_tokens)
    }


print("\nCalculating token lengths...")

combined = combined.map(
    get_token_counts
)


# ------------------------------------------------------------
# TOKEN STATISTICS
# ------------------------------------------------------------

output_lengths = combined[
    "output_token_count"
]

total_lengths = combined[
    "total_token_count"
]


print("\nTOKEN STATISTICS")
print("-" * 70)


print(
    f"Output tokens - min: {min(output_lengths)}, "
    f"max: {max(output_lengths)}, "
    f"average: "
    f"{sum(output_lengths) / len(output_lengths):.2f}"
)


print(
    f"Total tokens  - min: {min(total_lengths)}, "
    f"max: {max(total_lengths)}, "
    f"average: "
    f"{sum(total_lengths) / len(total_lengths):.2f}"
)


# ============================================================
# APPLY TOKEN FILTERS
# ============================================================

short_output_count = sum(
    1
    for x in combined["output_token_count"]
    if x < MIN_OUTPUT_TOKENS
)


long_sequence_count = sum(
    1
    for x in combined["total_token_count"]
    if x > MAX_TOTAL_TOKENS
)


combined = combined.filter(
    lambda x:
        x["output_token_count"]
        >= MIN_OUTPUT_TOKENS
        and
        x["total_token_count"]
        <= MAX_TOTAL_TOKENS
)


print(
    f"\nRemoved short outputs (<20 tokens): "
    f"{short_output_count}"
)


print(
    f"Removed long sequences (>2048 tokens): "
    f"{long_sequence_count}"
)


print(
    f"Remaining rows: {len(combined)}"
)


# ============================================================
# REMOVE TOKEN COUNT COLUMNS
# ============================================================

combined = combined.remove_columns(
    [
        "output_token_count",
        "total_token_count"
    ]
)


# ============================================================
# STEP 8 — TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 — TRAIN / VALIDATION SPLIT")
print("=" * 70)


split_dataset = combined.train_test_split(
    test_size=0.10,
    seed=RANDOM_SEED
)


train_dataset = split_dataset["train"]

validation_dataset = split_dataset["test"]


print(
    f"Training examples   : "
    f"{len(train_dataset)}"
)


print(
    f"Validation examples : "
    f"{len(validation_dataset)}"
)


# ============================================================
# STEP 9 — SAVE PARQUET
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 — SAVING PARQUET FILES")
print("=" * 70)


train_parquet = (
    OUTPUT_DIR / "train.parquet"
)

validation_parquet = (
    OUTPUT_DIR / "validation.parquet"
)


train_dataset.to_parquet(
    str(train_parquet)
)


validation_dataset.to_parquet(
    str(validation_parquet)
)


print(
    f"Saved: {train_parquet}"
)

print(
    f"Saved: {validation_parquet}"
)


# ============================================================
# STEP 10 — SAVE JSONL
# ============================================================

print("\n" + "=" * 70)
print("STEP 10 — SAVING JSONL FILES")
print("=" * 70)


train_jsonl = (
    OUTPUT_DIR / "train.jsonl"
)

validation_jsonl = (
    OUTPUT_DIR / "validation.jsonl"
)


def save_jsonl(dataset, path):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for example in dataset:

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                )
                + "\n"
            )


save_jsonl(
    train_dataset,
    train_jsonl
)


save_jsonl(
    validation_dataset,
    validation_jsonl
)


print(
    f"Saved: {train_jsonl}"
)

print(
    f"Saved: {validation_jsonl}"
)


# ============================================================
# STEP 11 — SAVE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 11 — SAVING PREPROCESSING SUMMARY")
print("=" * 70)


summary = {

    "datasets": {

        "indian_legal_v3":
            len(v3),

        "indian_legal_sft":
            len(legal_sft),

        "bns_bnss_bsa":
            len(bns),

        "constitution":
            len(constitution)
    },

    "combined_before_filtering":
        224377,

    "removed_empty_outputs":
        removed_empty,

    "removed_duplicates":
        removed_duplicates,

    "removed_short_outputs":
        short_output_count,

    "removed_long_sequences":
        long_sequence_count,

    "final_total":
        len(combined),

    "train_examples":
        len(train_dataset),

    "validation_examples":
        len(validation_dataset),

    "validation_split":
        0.10,

    "random_seed":
        RANDOM_SEED,

    "tokenizer":
        TOKENIZER_NAME,

    "min_output_tokens":
        MIN_OUTPUT_TOKENS,

    "max_total_tokens":
        MAX_TOTAL_TOKENS
}


summary_path = (
    OUTPUT_DIR /
    "preprocessing_summary.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    f"Saved: {summary_path}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 PREPROCESSING COMPLETE")
print("=" * 70)


print("\nFinal dataset sizes:")


print(
    f"  Train       : "
    f"{len(train_dataset)}"
)


print(
    f"  Validation  : "
    f"{len(validation_dataset)}"
)


print(
    f"  Total       : "
    f"{len(combined)}"
)


print("\nOutput directory:")

print(
    OUTPUT_DIR
)


print(
    "\nStage 2 preprocessing finished successfully."
)