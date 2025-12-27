"""
KernelBench Evaluator for OpenEvolve
====================================

Minimal integration that calls KernelBench's eval_kernel_against_ref()
to evaluate CUDA kernel optimizations.

This is Option A (minimal viable integration) - see ACTION_ROADMAP.md for details.
"""

import importlib.util
import os
import sys
import traceback
import numpy as np

# Add KernelBench to path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
KERNELBENCH_PATH = os.path.join(WORKSPACE_ROOT, "KernelBench")
if KERNELBENCH_PATH not in sys.path:
    sys.path.insert(0, KERNELBENCH_PATH)

from src.eval import eval_kernel_against_ref, KernelExecResult
from src.utils import read_file


# ============================================================================
# Configuration - Change these for different problems
# ============================================================================

# Default problem configuration
DEFAULT_LEVEL = 3
DEFAULT_PROBLEM_ID = 43

# Baseline timing (in ms) - measured empirically for each problem
# This is used to calculate speedup
BASELINE_TIMES = {
    # Level 3, Problem 43: MinGPT Causal Attention
    (3, 43): 34.9,  # PyTorch eager mode baseline (ms)
}


def get_reference_model_src(level: int, problem_id: int) -> str:
    """
    Load reference model source code from KernelBench problem files.
    
    Args:
        level: Problem level (1-4)
        problem_id: Problem ID within the level
        
    Returns:
        Source code string of the reference model
    """
    level_dir = os.path.join(KERNELBENCH_PATH, "KernelBench", f"level{level}")
    
    # Find the problem file
    for filename in os.listdir(level_dir):
        if filename.endswith(".py"):
            # Extract problem ID from filename (e.g., "43_MinGPTCausalAttention.py")
            try:
                file_problem_id = int(filename.split("_")[0])
                if file_problem_id == problem_id:
                    problem_path = os.path.join(level_dir, filename)
                    return read_file(problem_path)
            except (ValueError, IndexError):
                continue
    
    raise FileNotFoundError(f"Problem {problem_id} not found in level{level}")


def evaluate(program_path: str) -> dict:
    """
    Evaluate a CUDA kernel optimization against the KernelBench reference.
    
    This is the main entry point called by OpenEvolve.
    
    Args:
        program_path: Path to the generated kernel file (must define ModelNew class)
        
    Returns:
        Dictionary of metrics including combined_score for evolution
    """
    # Get problem configuration from environment or use defaults
    level = int(os.environ.get("KERNELBENCH_LEVEL", DEFAULT_LEVEL))
    problem_id = int(os.environ.get("KERNELBENCH_PROBLEM_ID", DEFAULT_PROBLEM_ID))
    
    print(f"[KernelBench Evaluator] Level {level}, Problem {problem_id}")
    
    try:
        # Load reference model source
        ref_src = get_reference_model_src(level, problem_id)
        print(f"[KernelBench Evaluator] Loaded reference model")
        
        # Load custom kernel source
        with open(program_path, "r") as f:
            kernel_src = f.read()
        print(f"[KernelBench Evaluator] Loaded kernel from {program_path}")
        
        # Check if ModelNew class exists in kernel
        if "class ModelNew" not in kernel_src:
            print("[KernelBench Evaluator] ERROR: Missing 'class ModelNew'")
            return {
                "combined_score": 0.0,
                "compiled": 0.0,
                "correct": 0.0,
                "speedup": 0.0,
                "error": "Missing ModelNew class",
            }
        
        # Run KernelBench evaluation
        print("[KernelBench Evaluator] Running evaluation...")
        result: KernelExecResult = eval_kernel_against_ref(
            original_model_src=ref_src,
            custom_model_src=kernel_src,
            num_correct_trials=5,  # Multiple trials for reliability
            num_perf_trials=10,
            measure_performance=True,
            verbose=True,
        )
        
        # Handle compilation failure
        if not result.compiled:
            print("[KernelBench Evaluator] FAIL: Compilation failed")
            error_msg = result.metadata.get("compilation_error", "Unknown compilation error")
            return {
                "combined_score": 0.0,
                "compiled": 0.0,
                "correct": 0.0,
                "speedup": 0.0,
                "error": str(error_msg)[:200],  # Truncate for LLM context
            }
        
        # Handle correctness failure (HARD GATE - no partial credit)
        if not result.correctness:
            print("[KernelBench Evaluator] FAIL: Correctness check failed")
            error_msg = result.metadata.get("correctness_issue", "Output mismatch")
            return {
                "combined_score": 0.0,  # Zero score for incorrect kernels
                "compiled": 1.0,
                "correct": 0.0,
                "speedup": 0.0,
                "error": str(error_msg)[:200],
            }
        
        # Calculate speedup
        baseline_time = BASELINE_TIMES.get((level, problem_id), 100.0)
        kernel_time = result.runtime  # in ms
        
        if kernel_time > 0:
            speedup = baseline_time / kernel_time
        else:
            speedup = 1.0
        
        # Calculate combined score
        # Note: This is Option A minimal scoring - can be enhanced later
        # Speedup is capped at 10x to prevent reward hacking
        capped_speedup = min(speedup, 10.0)
        combined_score = capped_speedup
        
        print(f"[KernelBench Evaluator] SUCCESS!")
        print(f"  Baseline: {baseline_time:.2f} ms")
        print(f"  Kernel:   {kernel_time:.2f} ms")
        print(f"  Speedup:  {speedup:.2f}x")
        print(f"  Score:    {combined_score:.2f}")
        
        return {
            "combined_score": float(combined_score),
            "compiled": 1.0,
            "correct": 1.0,
            "speedup": float(speedup),
            "runtime_ms": float(kernel_time),
            "baseline_ms": float(baseline_time),
        }
        
    except FileNotFoundError as e:
        print(f"[KernelBench Evaluator] ERROR: {e}")
        return {
            "combined_score": 0.0,
            "compiled": 0.0,
            "correct": 0.0,
            "speedup": 0.0,
            "error": str(e),
        }
    except Exception as e:
        print(f"[KernelBench Evaluator] ERROR: {e}")
        traceback.print_exc()
        return {
            "combined_score": 0.0,
            "compiled": 0.0,
            "correct": 0.0,
            "speedup": 0.0,
            "error": str(e)[:200],
        }


# ============================================================================
# Cascade Evaluation (Optional - for future enhancement)
# ============================================================================

def evaluate_stage1(program_path: str) -> dict:
    """
    Quick validation stage - just check if it compiles.
    """
    level = int(os.environ.get("KERNELBENCH_LEVEL", DEFAULT_LEVEL))
    problem_id = int(os.environ.get("KERNELBENCH_PROBLEM_ID", DEFAULT_PROBLEM_ID))
    
    try:
        ref_src = get_reference_model_src(level, problem_id)
        
        with open(program_path, "r") as f:
            kernel_src = f.read()
        
        if "class ModelNew" not in kernel_src:
            return {"combined_score": 0.0, "stage1_passed": 0.0}
        
        # Quick eval - just check compilation and 1 correctness trial
        result = eval_kernel_against_ref(
            original_model_src=ref_src,
            custom_model_src=kernel_src,
            num_correct_trials=1,
            measure_performance=False,
            verbose=False,
        )
        
        if result.compiled and result.correctness:
            return {"combined_score": 1.0, "stage1_passed": 1.0}
        elif result.compiled:
            return {"combined_score": 0.3, "stage1_passed": 0.5}
        else:
            return {"combined_score": 0.0, "stage1_passed": 0.0}
            
    except Exception as e:
        return {"combined_score": 0.0, "stage1_passed": 0.0, "error": str(e)}


def evaluate_stage2(program_path: str) -> dict:
    """
    Full evaluation stage - correctness + performance.
    """
    return evaluate(program_path)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        # Default test with our generated kernel
        test_path = os.path.join(
            WORKSPACE_ROOT, 
            "experiments", 
            "generated_kernels", 
            "level3_problem43_v1.py"
        )
    
    print(f"Testing evaluator with: {test_path}")
    result = evaluate(test_path)
    print(f"\nResult: {result}")

