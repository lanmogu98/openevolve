# KernelBench Integration for OpenEvolve

This example integrates [KernelBench](https://github.com/ScalingIntelligence/KernelBench) 
with OpenEvolve to automatically evolve optimized CUDA kernels.

## Overview

KernelBench is a benchmark for evaluating LLM's ability to write GPU kernels. 
This integration uses OpenEvolve to iteratively improve kernel implementations
through evolutionary optimization.

## Quick Start

### Prerequisites

1. NVIDIA GPU with CUDA support (tested on A800-SXM4-40GB)
2. PyTorch 2.0+ with CUDA
3. KernelBench installed (see parent project setup)

### Running

```bash
# Set the problem to optimize
export KERNELBENCH_LEVEL=3
export KERNELBENCH_PROBLEM_ID=43

# Run OpenEvolve
cd /path/to/kernelbench-openevolve/openevolve
python openevolve-run.py \
    --initial-program examples/kernelbench/initial_program.py \
    --evaluator examples/kernelbench/evaluator.py \
    --config examples/kernelbench/config.yaml \
    --max-iterations 50
```

## Files

| File | Description |
|------|-------------|
| `evaluator.py` | Calls KernelBench's `eval_kernel_against_ref()` |
| `initial_program.py` | Starting point for evolution (baseline implementation) |
| `config.yaml` | OpenEvolve configuration (LLM, database, evaluator settings) |

## Configuration

### Changing the Problem

Modify environment variables or edit `evaluator.py`:

```python
# In evaluator.py
DEFAULT_LEVEL = 3
DEFAULT_PROBLEM_ID = 43
```

Available problems: See `KernelBench/KernelBench/level{1,2,3,4}/`

### Baseline Times

Add baseline timing for new problems in `evaluator.py`:

```python
BASELINE_TIMES = {
    (3, 43): 34.9,  # Level 3, Problem 43
    (1, 1): 50.0,   # Level 1, Problem 1 (example)
}
```

## Evaluation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  OpenEvolve                                                     │
│  ─────────                                                      │
│  1. Generate candidate kernel (LLM)                             │
│  2. Write to temp file                                          │
│  3. Call evaluator.py                                           │
│       └── evaluator.py calls KernelBench eval                   │
│           ├── Compile CUDA code                                 │
│           ├── Check correctness (vs reference)                  │
│           └── Measure performance                               │
│  4. Return combined_score (speedup)                             │
│  5. Update population, repeat                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Metrics

| Metric | Description |
|--------|-------------|
| `combined_score` | Main evolution target (capped speedup) |
| `compiled` | 1.0 if CUDA compilation succeeded |
| `correct` | 1.0 if output matches reference |
| `speedup` | kernel_time / baseline_time |
| `runtime_ms` | Kernel execution time in milliseconds |

## Limitations (Option A - Minimal Integration)

This is a minimal integration with known limitations:

1. **Single shape validation** - May be susceptible to hardcoding
2. **No Dev/Holdout split** - No generalization testing
3. **Simple speedup scoring** - Could be reward-hacked

See `mynotes/ACTION_ROADMAP.md` for planned enhancements (Option B).

## Troubleshooting

### HuggingFace Access Issues

On servers without direct HuggingFace access:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### CUDA Compilation Errors

Check that:
- CUDA toolkit matches PyTorch CUDA version
- `ninja` is installed for parallel compilation
- Sufficient disk space in temp directories

