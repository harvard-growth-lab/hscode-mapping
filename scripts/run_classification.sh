#!/bin/bash
#SBATCH --job-name=hs-classify
#SBATCH --output=slurm_logs/classify_%j.log
#SBATCH --error=slurm_logs/classify_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --partition=shared
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kdaryanani@hks.harvard.edu

# Use lab directory for HuggingFace model cache (home dir has quota limits)
export HF_HOME=/n/holylfs05/LABS/hausmann_lab/Lab/kdaryanani/.cache/huggingface

cd /n/holylfs05/LABS/hausmann_lab/Lab/kdaryanani/linkages-pkg
source .venv/bin/activate

uv run python pipeline.py --checkpoint-path data/intermediate/checkpoint.parquet
