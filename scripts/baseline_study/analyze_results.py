#!/usr/bin/env python3
"""Analyze baseline study results and generate paper figures.

Reads results.json, computes statistics, generates comparison plots.

Usage:
    python scripts/baseline_study/analyze_results.py
    python scripts/baseline_study/analyze_results.py --results path/to/results.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Try matplotlib with LaTeX
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.5,
    })
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_results(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data.get("runs", [])


def compute_stats(runs: list[dict]) -> dict:
    """Compute per-task, per-condition statistics."""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r.get("tokens_total") is None:
            continue
        key = (r["task_id"], r["task_name"])
        cond = r["condition"]
        grouped[key][cond].append(r)

    stats = {}
    for (task_id, task_name), conditions in sorted(grouped.items()):
        task_stats = {"task_id": task_id, "task_name": task_name}
        for cond in ["with_mcp", "without_mcp"]:
            runs_c = conditions.get(cond, [])
            if not runs_c:
                continue
            tokens = [r["tokens_total"] for r in runs_c]
            times = [r["wall_clock_seconds"] for r in runs_c
                     if r.get("wall_clock_seconds")]
            attempts = [r["simulation_attempts"] for r in runs_c
                        if r.get("simulation_attempts")]
            first_success = [r["first_attempt_success"] for r in runs_c
                             if r.get("first_attempt_success") is not None]

            task_stats[cond] = {
                "tokens_median": int(np.median(tokens)),
                "tokens_mean": int(np.mean(tokens)),
                "tokens_min": int(min(tokens)),
                "tokens_max": int(max(tokens)),
                "time_median": round(np.median(times), 1) if times else None,
                "attempts_median": round(np.median(attempts), 1) if attempts else None,
                "first_success_rate": round(sum(first_success) / len(first_success), 2)
                    if first_success else None,
                "n_runs": len(runs_c),
            }

        # Compute savings
        if "with_mcp" in task_stats and "without_mcp" in task_stats:
            wm = task_stats["with_mcp"]["tokens_median"]
            wo = task_stats["without_mcp"]["tokens_median"]
            task_stats["token_savings_pct"] = round((1 - wm / wo) * 100, 1) if wo > 0 else 0
            task_stats["token_ratio"] = round(wm / wo, 2) if wo > 0 else 1.0

        stats[task_id] = task_stats
    return stats


def print_summary(stats: dict):
    """Print a human-readable summary table."""
    print("\n" + "=" * 90)
    print("BASELINE STUDY RESULTS")
    print("=" * 90)
    print(f"{'Task':<35} {'Without MCP':>12} {'With MCP':>10} {'Savings':>10} {'Attempts':>10}")
    print("-" * 90)

    for tid, s in sorted(stats.items()):
        name = s["task_name"][:33]
        wo = s.get("without_mcp", {}).get("tokens_median", "?")
        wm = s.get("with_mcp", {}).get("tokens_median", "?")
        savings = f"{s.get('token_savings_pct', '?')}%"
        att_wo = s.get("without_mcp", {}).get("attempts_median", "?")
        att_wm = s.get("with_mcp", {}).get("attempts_median", "?")
        print(f"{name:<35} {wo:>12} {wm:>10} {savings:>10} {att_wo}/{att_wm}")

    print("=" * 90)


def plot_token_comparison(stats: dict, output_dir: Path):
    """Generate the main comparison figure for the paper."""
    if not HAS_MPL:
        print("matplotlib not available — skipping plot")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    mm = 1.0 / 25.4

    tasks = sorted(stats.values(), key=lambda s: s["task_id"])
    task_names = [s["task_name"] for s in tasks]
    n = len(tasks)

    wo_tokens = [s.get("without_mcp", {}).get("tokens_median", 0) for s in tasks]
    wm_tokens = [s.get("with_mcp", {}).get("tokens_median", 0) for s in tasks]

    # ── Figure 1: Token comparison bar chart ──
    fig, ax = plt.subplots(figsize=(180*mm, 75*mm))
    x = np.arange(n)
    w = 0.35
    bars1 = ax.bar(x - w/2, wo_tokens, w, label="Without MCP", color="#EF4444", alpha=0.8)
    bars2 = ax.bar(x + w/2, wm_tokens, w, label="With MCP", color="#16A34A", alpha=0.8)

    ax.set_ylabel("Total tokens (median of 3 runs)")
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=25, ha="right", fontsize=6)
    ax.legend(frameon=True, fancybox=False, edgecolor="#ccc", framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add savings percentage labels
    for i, (wo, wm) in enumerate(zip(wo_tokens, wm_tokens)):
        if wo > 0 and wm > 0:
            pct = round((1 - wm / wo) * 100)
            ax.text(i, max(wo, wm) * 1.05, f"$-{pct}\\%$",
                    ha="center", fontsize=6, color="#166534")

    fig.savefig(output_dir / "token_comparison.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_dir / "token_comparison.svg", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {output_dir / 'token_comparison.pdf'}")

    # ── Figure 2: Savings ratio vs difficulty ──
    fig, ax = plt.subplots(figsize=(90*mm, 70*mm))
    savings = [s.get("token_savings_pct", 0) for s in tasks]
    colors = {"easy": "#60A5FA", "medium": "#FBBF24", "hard": "#EF4444"}
    difficulties = ["easy", "easy", "medium", "medium", "hard", "hard"]
    bar_colors = [colors.get(d, "#999") for d in difficulties]

    ax.bar(x, savings, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Token savings (\\%)")
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=25, ha="right", fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 100)

    # Legend for difficulty
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[d], label=d.capitalize())
                       for d in ["easy", "medium", "hard"]]
    ax.legend(handles=legend_elements, frameon=True, fancybox=False,
              edgecolor="#ccc", framealpha=0.95)

    fig.savefig(output_dir / "savings_by_difficulty.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_dir / "savings_by_difficulty.svg", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {output_dir / 'savings_by_difficulty.pdf'}")

    # ── Figure 3: Attempts comparison ──
    wo_att = [s.get("without_mcp", {}).get("attempts_median", 0) for s in tasks]
    wm_att = [s.get("with_mcp", {}).get("attempts_median", 0) for s in tasks]

    if any(a > 0 for a in wo_att + wm_att):
        fig, ax = plt.subplots(figsize=(90*mm, 70*mm))
        ax.bar(x - w/2, wo_att, w, label="Without MCP", color="#EF4444", alpha=0.8)
        ax.bar(x + w/2, wm_att, w, label="With MCP", color="#16A34A", alpha=0.8)
        ax.set_ylabel("Simulation attempts (median)")
        ax.set_xticks(x)
        ax.set_xticklabels(task_names, rotation=25, ha="right", fontsize=6)
        ax.legend(frameon=True, fancybox=False, edgecolor="#ccc", framealpha=0.95)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.savefig(output_dir / "attempts_comparison.pdf", bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        print(f"Saved: {output_dir / 'attempts_comparison.pdf'}")


def generate_latex_table(stats: dict) -> str:
    """Generate a LaTeX table for the paper."""
    lines = [
        r"\begin{table}[htbp!]",
        r"  \centering",
        r"  \caption{Token consumption comparison: with and without the MCP knowledge base.}",
        r"  \label{tab:token_savings}",
        r"  \small",
        r"  \begin{tabular}{llrrrl}",
        r"    \toprule",
        r"    Difficulty & Task & Without MCP & With MCP & Savings & Attempts \\",
        r"    \midrule",
    ]

    difficulties = {1: "Easy", 2: "Easy", 3: "Medium", 4: "Medium", 5: "Hard", 6: "Hard"}
    for tid, s in sorted(stats.items()):
        diff = difficulties.get(tid, "?")
        name = s["task_name"]
        wo = s.get("without_mcp", {}).get("tokens_median", "---")
        wm = s.get("with_mcp", {}).get("tokens_median", "---")
        savings = f"{s.get('token_savings_pct', '?')}\\%"
        att_wo = s.get("without_mcp", {}).get("attempts_median", "?")
        att_wm = s.get("with_mcp", {}).get("attempts_median", "?")
        wo_fmt = f"{wo:,}" if isinstance(wo, int) else wo
        wm_fmt = f"{wm:,}" if isinstance(wm, int) else wm
        lines.append(f"    {diff} & {name} & {wo_fmt} & {wm_fmt} & {savings} & {att_wo}/{att_wm} \\\\")

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze baseline study results")
    parser.add_argument("--results", type=str,
                        default="scripts/baseline_study/results.json",
                        help="Path to results JSON file")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Copy results_template.json to results.json and fill in data.")
        sys.exit(1)

    runs = load_results(results_path)
    if not runs:
        print("No results found in file.")
        sys.exit(1)

    stats = compute_stats(runs)
    print_summary(stats)

    output_dir = Path("scripts/baseline_study/figures")
    plot_token_comparison(stats, output_dir)

    latex = generate_latex_table(stats)
    (output_dir / "token_table.tex").write_text(latex)
    print(f"Saved: {output_dir / 'token_table.tex'}")


if __name__ == "__main__":
    main()
