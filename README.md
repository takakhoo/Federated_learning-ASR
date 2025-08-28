# Federated Learning: ASR Gradient Reconstruction

**Author:** Taka Khoo  
**Collaborator:** Minh Bui  
**Date:** August 27, 2024  

## Project Overview

This project implements federated learning for ASR gradient reconstruction, focusing on two main datasets:
- **DS1**: DeepSpeech1 dataset (working baseline)
- **DS2**: DeepSpeech2 dataset (needs debugging and optimization)

## Project Structure

```
federated_learning/
├── asr-grad-reconstruction/    # Main repository (copied from Minh)
│   ├── src/                    # Source code
│   │   ├── reconstruct_ds1_run_many_sample.py  # DS1 training script
│   │   ├── reconstruct_ds2_run_many_sample.py  # DS2 training script
│   │   ├── models/             # Model implementations
│   │   ├── data/               # Data loading and preprocessing
│   │   └── utils/              # Utility functions
│   ├── environment.yml         # Conda environment
│   └── run.bash                # Example run commands
├── DS1_runs/                   # DS1 experiment results and logs
├── DS2_runs/                   # DS2 experiment results and debugging logs
└── paper_notes/                # Notes for ICASSP paper
```

## Setup

### 1. Environment Setup
```bash
# Navigate to the project directory
cd /scratch2/f004h1v/federated_learning/asr-grad-reconstruction

# Create and activate conda environment
conda env create -f environment.yml -n fl-env
conda activate fl-env
```

### 2. Verify Installation
```bash
# Check if PyTorch is available
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check GPU availability
nvidia-smi
```

## Running Experiments

### DS1: Baseline Experiment (Working)

The DS1 experiment is the working baseline that needs to be replicated to understand the pipeline.

#### Basic DS1 Run
```bash
cd /scratch2/f004h1v/federated_learning/asr-grad-reconstruction

# Activate environment
conda activate fl-env

# Run DS1 with default parameters
CUDA_VISIBLE_DEVICES="0" python src/reconstruct_ds1_run_many_sample.py \
    --batch-start 0 \
    --batch-end 100 \
    --batch_min_dur 0 \
    --batch_max_dur 1000 \
    --lr 0.5 \
    --max_iter 2000 \
    --n_seeds 10
```

#### DS1 with Different Learning Rates
```bash
# Learning rate experiments
CUDA_VISIBLE_DEVICES="0" python src/reconstruct_ds1_run_many_sample.py \
    --batch-start 0 --batch-end 100 \
    --batch_min_dur 1000 --batch_max_dur 2000 \
    --lr 0.1 --max_iter 2000

CUDA_VISIBLE_DEVICES="1" python src/reconstruct_ds1_run_many_sample.py \
    --batch-start 0 --batch-end 100 \
    --batch_min_dur 1000 --batch_max_dur 2000 \
    --lr 0.01 --max_iter 2000

CUDA_VISIBLE_DEVICES="2" python src/reconstruct_ds1_run_many_sample.py \
    --batch-start 0 --batch-end 100 \
    --batch_min_dur 1000 --batch_max_dur 2000 \
    --lr 0.001 --max_iter 2000
```

#### DS1 with Gradient Percentage Experiments
```bash
# Top gradient percentage experiments
CUDA_VISIBLE_DEVICES="0" python src/reconstruct_ds1_run_many_sample.py \
    --batch-start 0 --batch-end 100 \
    --batch_min_dur 1000 --batch_max_dur 2000 \
    --lr 0.5 --top_grad_percentage 0.2 --max_iter 2000

CUDA_VISIBLE_DEVICES="1" python src/reconstruct_ds1_run_many_sample.py \
    --batch-start 0 --batch-end 100 \
    --batch_min_dur 1000 --batch_max_dur 2000 \
    --lr 0.5 --top_grad_percentage 0.4 --max_iter 2000

CUDA_VISIBLE_DEVICES="2" python src/reconstruct_ds1_run_many_sample.py \
    --batch-start 0 --batch-end 100 \
    --batch_min_dur 1000 --batch_max_dur 2000 \
    --lr 0.5 --top_grad_percentage 0.6 --max_iter 2000
```

### DS2: Debugging and Optimization

The DS2 experiment is currently failing and needs debugging. The goal is to identify why it's not working and fix the issues.

#### Basic DS2 Run (Debug Mode)
```bash
cd /scratch2/f004h1v/federated_learning/asr-grad-reconstruction

# Activate environment
conda activate fl-env

# Run DS2 with verbose logging
CUDA_VISIBLE_DEVICES="0" python src/reconstruct_ds2_run_many_sample.py \
    --batch-start 0 \
    --batch-end 50 \
    --batch_min_dur 0 \
    --batch_max_dur 500 \
    --lr 0.5 \
    --max_iter 1000 \
    --n_seeds 5
```

#### DS2 Parameter Tuning
```bash
# Try different learning rates
CUDA_VISIBLE_DEVICES="0" python src/reconstruct_ds2_run_many_sample.py \
    --batch-start 0 --batch-end 50 \
    --batch_min_dur 0 --batch_max_dur 500 \
    --lr 0.1 --max_iter 1000

CUDA_VISIBLE_DEVICES="1" python src/reconstruct_ds2_run_many_sample.py \
    --batch-start 0 --batch-end 50 \
    --batch_min_dur 0 --batch_max_dur 500 \
    --lr 0.01 --max_iter 1000

# Try different batch sizes
CUDA_VISIBLE_DEVICES="2" python src/reconstruct_ds2_run_many_sample.py \
    --batch-start 0 --batch-end 25 \
    --batch_min_dur 0 --batch_max_dur 250 \
    --lr 0.5 --max_iter 1000

# Try different optimization parameters
CUDA_VISIBLE_DEVICES="3" python src/reconstruct_ds2_run_many_sample.py \
    --batch-start 0 --batch-end 50 \
    --batch_min_dur 0 --batch_max_dur 500 \
    --lr 0.5 --max_iter 500 --n_seeds 3
```

## Logging and Results

### Logging
All experiments should be logged with timestamps:

```bash
# Log DS1 experiments
CUDA_VISIBLE_DEVICES="0" python src/reconstruct_ds1_run_many_sample.py \
    --batch-start 0 --batch-end 100 \
    --batch_min_dur 1000 --batch_max_dur 2000 \
    --lr 0.5 --max_iter 2000 \
    2>&1 | tee /scratch2/f004h1v/federated_learning/DS1_runs/ds1_lr05_$(date +%F_%H-%M).log

# Log DS2 experiments
CUDA_VISIBLE_DEVICES="0" python src/reconstruct_ds2_run_many_sample.py \
    --batch-start 0 --batch-end 50 \
    --batch_min_dur 0 --batch_max_dur 500 \
    --lr 0.5 --max_iter 1000 \
    2>&1 | tee /scratch2/f004h1v/federated_learning/DS2_runs/ds2_debug_$(date +%F_%H-%M).log
```

### Results Structure
```
DS1_runs/
├── ds1_lr05_2024-08-27_23-00.log
├── ds1_lr01_2024-08-27_23-30.log
├── ds1_grad20_2024-08-27_23-45.log
└── results/
    ├── checkpoints/
    ├── plots/
    └── metrics.csv

DS2_runs/
├── ds2_debug_2024-08-27_23-15.log
├── ds2_lr01_2024-08-27_23-45.log
├── ds2_batch25_2024-08-28_00-00.log
└── results/
    ├── checkpoints/
    ├── plots/
    └── metrics.csv
```

## Key Parameters to Monitor

### DS1 (Working Baseline)
- **Learning Rate**: 0.5 (default), 0.1, 0.01, 0.001
- **Batch Duration**: 0-1000, 1000-2000, 2000-3000, 3000-4000
- **Gradient Percentage**: 100%, 90%, 80%, 70%, 60%, 50%
- **Distance Function**: cosine, L2, cosine+L2

### DS2 (Debugging)
- **Learning Rate**: Start with 0.1, 0.01 (lower than DS1)
- **Batch Size**: Reduce from 100 to 25-50
- **Max Iterations**: Start with 500-1000 (lower than DS1)
- **Seeds**: Start with 3-5 (lower than DS1)

## Debugging DS2 Issues

### Common Problems to Check
1. **Dataset Loading**: Are the DS2 files accessible?
2. **Memory Issues**: Does DS2 require more memory than DS1?
3. **Model Compatibility**: Are the DS1/DS2 models compatible?
4. **Hyperparameter Sensitivity**: Is DS2 more sensitive to learning rate?

### Debugging Steps
1. **Run with minimal parameters** (small batch, low iterations)
2. **Check error messages** in logs
3. **Compare model architectures** between DS1 and DS2
4. **Monitor GPU memory usage** during training
5. **Try different optimization strategies**

## ICASSP Paper Preparation

### Structure
1. **Introduction**: FL for speech/asr gradient reconstruction
2. **Related Work**: Federated learning, speech processing
3. **Method**: System pipeline, DS1/DS2 configurations
4. **Experiments**: DS1 reproduction, DS2 debugging results
5. **Discussion**: Why DS2 fails, insights gained
6. **Conclusion**: Future work, codec connections

### Key Results to Document
- DS1 reproduction accuracy and convergence
- DS2 failure modes and debugging insights
- Hyperparameter sensitivity analysis
- Computational cost comparisons
- Lessons learned for future FL implementations

## Next Steps

1. **Replicate DS1 experiments** to understand the pipeline
2. **Debug DS2 systematically** with parameter sweeps
3. **Document all results** in structured format
4. **Prepare ICASSP paper outline** with concrete results
5. **Coordinate with Minh** on next meeting (Thursday)

## Contact

- **Taka Khoo**: f004h1v@dartmouth.edu
- **Minh Bui**: [Contact information]

---

**Note:** This project is part of ongoing research on federated learning for speech processing and will contribute to the ICASSP paper submission.
