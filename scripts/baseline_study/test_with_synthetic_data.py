#!/usr/bin/env python3
"""Generate synthetic results and test the analysis pipeline.

This validates that the plotting and statistics code works before
real data is collected.
"""

import json
import random
from pathlib import Path

# Synthetic data based on expected results
SYNTHETIC_RUNS = []

TASKS = [
    (1, "Poisson unit square", "easy", 3000, 2500),
    (2, "Linear elasticity cantilever", "easy", 5000, 3000),
    (3, "Vortex shedding Re=100", "medium", 15000, 5500),
    (4, "h-convergence study", "medium", 10000, 4000),
    (5, "FSI flexible flag", "hard", 40000, 8000),
    (6, "Cross-solver plate-with-hole", "hard", 50000, 12000),
]

random.seed(42)

for task_id, name, diff, wo_base, wm_base in TASKS:
    for run in range(1, 4):
        # Without MCP
        wo_tokens = int(wo_base * random.uniform(0.85, 1.15))
        wo_time = wo_tokens / 50  # rough: 50 tokens/second
        wo_attempts = {"easy": 1, "medium": random.choice([2, 3]),
                       "hard": random.choice([4, 5, 6, 7])}[diff]
        SYNTHETIC_RUNS.append({
            "task_id": task_id, "task_name": name,
            "condition": "without_mcp", "run": run,
            "tokens_input": int(wo_tokens * 0.6),
            "tokens_output": int(wo_tokens * 0.4),
            "tokens_total": wo_tokens,
            "wall_clock_seconds": round(wo_time, 1),
            "simulation_attempts": wo_attempts,
            "first_attempt_success": diff == "easy",
            "final_error_percent": round(random.uniform(0.1, 5.0), 2),
            "notes": "synthetic"
        })
        # With MCP
        wm_tokens = int(wm_base * random.uniform(0.85, 1.15))
        wm_time = wm_tokens / 50
        wm_attempts = {"easy": 1, "medium": random.choice([1, 2]),
                       "hard": random.choice([1, 2, 3])}[diff]
        SYNTHETIC_RUNS.append({
            "task_id": task_id, "task_name": name,
            "condition": "with_mcp", "run": run,
            "tokens_input": int(wm_tokens * 0.6),
            "tokens_output": int(wm_tokens * 0.4),
            "tokens_total": wm_tokens,
            "wall_clock_seconds": round(wm_time, 1),
            "simulation_attempts": wm_attempts,
            "first_attempt_success": True,
            "final_error_percent": round(random.uniform(0.1, 5.0), 2),
            "notes": "synthetic"
        })

# Save and run analysis
out = Path(__file__).parent / "results.json"
out.write_text(json.dumps({"runs": SYNTHETIC_RUNS}, indent=2))
print(f"Wrote {len(SYNTHETIC_RUNS)} synthetic runs to {out}")

# Run the analysis
import subprocess, sys
result = subprocess.run(
    [sys.executable, str(Path(__file__).parent / "analyze_results.py"),
     "--results", str(out)],
    capture_output=True, text=True, cwd=str(Path(__file__).parent.parent.parent),
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-500:])
print(f"Exit code: {result.returncode}")
