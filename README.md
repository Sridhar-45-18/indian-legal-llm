# Indian Legal LLM Fine-Tuning Pipeline

A multi-stage data curation, preprocessing, and instruction fine-tuning pipeline designed for Indian Legal domain adaptation using modern open-source LLMs (e.g., Qwen 2.5).

---

## 📌 Overview

This repository provides an end-to-end framework to:
1. **Curate and Download** multi-source datasets covering Indian law (BNS, BNSS, BSA, Indian Legal QA, Constitution, LawInstruct, Indian Legal V3) and general instruction following (Alpaca).
2. **Standardize and Tokenize** datasets into staged curriculum training splits:
   - **Stage 1 (General Instruction)**: General reasoning and instruction tuning baseline.
   - **Stage 2 (Legal Core)**: In-depth Indian criminal and civil statutory understanding (Bharatiya Nyaya Sanhita, Bharatiya Nagarik Suraksha Sanhita, Bharatiya Sakshya Adhiniyam, and case law QA).
   - **Stage 3 (Enrichment)**: Constitutional analysis and specialized legal reasoning.
3. **Train & Fine-Tune**: Formatted outputs compatible with Hugging Face `transformers`, `trl` (SFTTrainer), `peft` (LoRA/QLoRA), and `unsloth`.

---

## 📁 Project Structure

```text
PREDECTIVE/
├── checkpoints/              # Model weights and checkpoint saves (gitignored)
├── data/                     # Raw and processed datasets (gitignored)
│   ├── raw/                  # Downloaded raw datasets
│   └── processed/            # Staged JSONL and Parquet datasets
│       ├── stage1_general/
│       ├── stage2_legal/
│       └── stage3_enrichment/
├── notebooks/                # Exploratory and prototyping Jupyter notebooks
├── src/
│   ├── data/                 # Data download, inspection, and preprocessing scripts
│   │   ├── download_alpaca.py
│   │   ├── download_bns_bnss_bsa.py
│   │   ├── download_constitution.py
│   │   ├── download_indian_legal_sft.py
│   │   ├── download_indian_legal_v3.py
│   │   ├── inspect_alpaca.py
│   │   ├── inspect_lawinstruct.py
│   │   ├── preprocess_alpaca.py
│   │   └── preprocess_stage2.py
│   └── training/             # Model training and fine-tuning scripts
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.10+ installed. A GPU with CUDA support is recommended for preprocessing tokenization and training.

### 2. Environment Setup

Clone this repository and set up a virtual environment:

```bash
git clone <YOUR_REPO_URL>
cd PREDECTIVE

# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Activate on Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🛠️ Data Pipeline Workflow

### Step 1: Download Datasets

Run the data acquisition scripts to download raw datasets from Hugging Face into `data/raw/`:

```bash
python src/data/download_alpaca.py
python src/data/download_bns_bnss_bsa.py
python src/data/download_constitution.py
python src/data/download_indian_legal_sft.py
python src/data/download_indian_legal_v3.py
```

### Step 2: Inspect Datasets

Verify token distributions and prompt length limits:

```bash
python src/data/inspect_alpaca.py
python src/data/inspect_lawinstruct.py
```

### Step 3: Run Preprocessing

Generate standardized train/validation splits in `.jsonl` and `.parquet` format:

```bash
# Stage 1: General Instruction Preprocessing
python src/data/preprocess_alpaca.py

# Stage 2: Indian Legal Preprocessing
python src/data/preprocess_stage2.py
```

Processed data will be saved to `data/processed/`.

---

## 🏋️ Training & Fine-Tuning

Training scripts utilize `unsloth`, `peft` (QLoRA 4-bit / LoRA), and `trl.SFTTrainer` for efficient training:
- Targets: Qwen 2.5 (1.5B / 7B) or Llama architectures.
- Checkpoints will be saved automatically to `checkpoints/`.

---

## 📄 License & Attribution

This project is licensed under the Apache 2.0 License / MIT License.
Please consult the respective licenses of upstream datasets (Hugging Face, LawInstruct, Indian Legal corpora) before commercial deployment.
