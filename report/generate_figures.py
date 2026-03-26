"""
Generate all figures for the Scaling AI Coding Agents on LUMI report.

Figures produced (saved to report/figures/):
  fig1_phase_pie.png         — Phase time distribution for a typical task
  fig2_phase_bars.png        — Stacked phase breakdown per task (2GPU serial)
  fig3_inference_vs_step.png — Inference time per step vs step number (all configs)
  fig4_config_comparison.png — Configuration comparison: tasks done, pass rate, model load

Usage:
    python3 report/generate_figures.py
Run from the repo root (researchProjectData/).
"""

import json
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Paths -----------------------------------------------------------------

RUNS = {
    "2GPU serial":      "experiments/lumi_glm_test_5/runs/runs_2gpu",
    "4GPU serial":      "experiments/lumi_glm_test_5/runs/runs_4gpu",
    "Parallel A (2GPU)": "experiments/lumi_glm_test_5/runs/runs_parallel_a",
    "Parallel B (2GPU)": "experiments/lumi_glm_test_5/runs/runs_parallel_b",
}

OUT_DIR = "report/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Colour palette (consistent across figures)
C_INF   = "#2166ac"   # blue  — inference
C_SETUP = "#4dac26"   # green — setup
C_EXEC  = "#d7191c"   # red   — exec
C_TEST  = "#fdae61"   # orange — test

# --- Data loading ----------------------------------------------------------

def load_metrics(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "quixbugs_*/metrics.json")))
    records = []
    for f in files:
        with open(f) as fh:
            records.append(json.load(fh))
    return records


def short_name(task_id):
    return task_id.replace("quixbugs_", "").replace("_", " ")


# ===========================================================================
# Figure 1 — Phase pie chart (representative task: bitcount, 2GPU serial)
# ===========================================================================

def fig1_phase_pie():
    records = load_metrics(RUNS["2GPU serial"])

    # Use aggregate across all normal tasks (exclude extreme outliers > 1800s wall)
    normal = [r for r in records if r["total_wall_time_s"] < 1800]

    total_inf   = sum(r["model_time_total_s"] for r in normal)
    total_setup = sum(r["setup_time_s"] for r in normal)
    total_exec  = sum(r["exec_time_total_s"] for r in normal)
    total_test  = sum(r["test_time_s"] for r in normal)
    total_wall  = sum(r["total_wall_time_s"] for r in normal)

    sizes  = [total_inf, total_setup, total_exec, total_test]
    labels = ["LLM inference", "Task setup", "Command exec", "Final test"]
    colors = [C_INF, C_SETUP, C_EXEC, C_TEST]
    explode = (0.03, 0.03, 0.03, 0.03)

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors, explode=explode,
        autopct=lambda p: f"{p:.1f}%" if p > 0.5 else "",
        startangle=140, pctdistance=0.75,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_color("white")
        at.set_fontweight("bold")

    ax.legend(wedges, labels, loc="lower right", fontsize=10,
              framealpha=0.9, edgecolor="#cccccc")
    ax.set_title(
        "Figure 1 — Pipeline phase time distribution\n"
        f"(2GPU serial, {len(normal)} normal tasks, total {total_wall/3600:.1f} GPU-h of task time)",
        fontsize=11, pad=14,
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig1_phase_pie.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# Figure 2 — Stacked bar: phase breakdown per task (2GPU serial)
# ===========================================================================

def fig2_phase_bars():
    records = load_metrics(RUNS["2GPU serial"])

    names   = [short_name(r["task_id"]) for r in records]
    inf_t   = [r["model_time_total_s"] for r in records]
    setup_t = [r["setup_time_s"] for r in records]
    exec_t  = [r["exec_time_total_s"] for r in records]
    test_t  = [r["test_time_s"] for r in records]
    passed  = [r.get("tests_passed", False) for r in records]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(14, 5))

    b1 = ax.bar(x, inf_t,   color=C_INF,   label="LLM inference", zorder=2)
    b2 = ax.bar(x, setup_t, bottom=inf_t,  color=C_SETUP, label="Setup", zorder=2)
    b3 = ax.bar(x, exec_t,
                bottom=[i + s for i, s in zip(inf_t, setup_t)],
                color=C_EXEC, label="Exec", zorder=2)
    b4 = ax.bar(x, test_t,
                bottom=[i + s + e for i, s, e in zip(inf_t, setup_t, exec_t)],
                color=C_TEST, label="Test", zorder=2)

    # Mark PASS tasks with a star above the bar
    for i, (ok, total) in enumerate(zip(passed, [i+s+e+t for i,s,e,t in zip(inf_t, setup_t, exec_t, test_t)])):
        if ok:
            ax.text(i, total + 20, "★", ha="center", va="bottom",
                    fontsize=8, color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7.5)
    ax.set_ylabel("Wall time (s)")
    ax.set_title(
        "Figure 2 — Per-task phase breakdown (2GPU serial, 24 tasks)\n"
        "★ = PASS   (tasks sorted by order of execution)",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}s"))
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig2_phase_bars.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# Figure 3 — Inference time per step vs step number (2GPU + parallel)
# ===========================================================================

def fig3_inference_vs_step():
    configs = {
        "2GPU serial":       (RUNS["2GPU serial"],       "#2166ac", "o"),
        "Parallel A+B":      (None,                      "#d7191c", "s"),
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    # 2GPU serial
    records_2gpu = load_metrics(RUNS["2GPU serial"])
    for r in records_2gpu:
        for entry in r.get("step_log", []):
            mt = entry.get("model_time_s", 0)
            if mt > 0:
                ax.scatter(entry["step"], mt, color="#2166ac", s=22,
                           alpha=0.65, marker="o", zorder=3)

    # Parallel (both batches combined)
    for run_dir in [RUNS["Parallel A (2GPU)"], RUNS["Parallel B (2GPU)"]]:
        for r in load_metrics(run_dir):
            for entry in r.get("step_log", []):
                mt = entry.get("model_time_s", 0)
                if mt > 0:
                    ax.scatter(entry["step"], mt, color="#d7191c", s=22,
                               alpha=0.55, marker="s", zorder=3)

    # Trend lines — median per step number for each config
    def median_trend(records, max_step=15):
        step_times = {s: [] for s in range(1, max_step + 1)}
        for r in records:
            for entry in r.get("step_log", []):
                mt = entry.get("model_time_s", 0)
                s = entry["step"]
                if mt > 0 and s <= max_step:
                    step_times[s].append(mt)
        xs = [s for s in range(1, max_step + 1) if step_times[s]]
        ys = [np.median(step_times[s]) for s in xs]
        return xs, ys

    xs2, ys2 = median_trend(records_2gpu)
    ax.plot(xs2, ys2, color="#2166ac", linewidth=2, linestyle="--",
            label="2GPU serial (median)", zorder=4)

    par_records = load_metrics(RUNS["Parallel A (2GPU)"]) + load_metrics(RUNS["Parallel B (2GPU)"])
    xsp, ysp = median_trend(par_records)
    ax.plot(xsp, ysp, color="#d7191c", linewidth=2, linestyle="--",
            label="2×2GPU parallel (median)", zorder=4)

    # Legend handles
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2166ac",
               markersize=7, label="2GPU serial (each step)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#d7191c",
               markersize=7, label="2×2GPU parallel (each step)"),
        Line2D([0], [0], color="#2166ac", linewidth=2, linestyle="--",
               label="2GPU serial (median)"),
        Line2D([0], [0], color="#d7191c", linewidth=2, linestyle="--",
               label="2×2GPU parallel (median)"),
    ]
    ax.legend(handles=handles, fontsize=9, framealpha=0.9)
    ax.set_xlabel("Step number in task", fontsize=10)
    ax.set_ylabel("Inference time (s)", fontsize=10)
    ax.set_title(
        "Figure 3 — Inference time per step vs step number\n"
        "(context growth increases KV cache size and slows each subsequent call)",
        fontsize=11,
    )
    ax.set_xlim(0.5, 15.5)
    ax.set_xticks(range(1, 16))
    ax.grid(alpha=0.3, zorder=0)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig3_inference_vs_step.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# Figure 4 — Configuration comparison bar chart
# ===========================================================================

def fig4_config_comparison():
    configs = [
        {
            "label":       "2GPU\nserial",
            "gpu_h":       16,
            "tasks_done":  24,
            "tasks_total": 40,
            "passed":      10,
            "model_load_min": 37.5,
            "color":       "#2166ac",
        },
        {
            "label":       "4GPU\nserial",
            "gpu_h":       32,
            "tasks_done":  26,
            "tasks_total": 40,
            "passed":      9,
            "model_load_min": 19.7,
            "color":       "#762a83",
        },
        {
            "label":       "2×2GPU\nparallel",
            "gpu_h":       32,
            "tasks_done":  40,
            "tasks_total": 40,
            "passed":      16,
            "model_load_min": 4.2,
            "color":       "#d7191c",
        },
    ]

    labels       = [c["label"] for c in configs]
    tasks_done   = [c["tasks_done"] for c in configs]
    pass_rates   = [100 * c["passed"] / c["tasks_done"] for c in configs]
    model_loads  = [c["model_load_min"] for c in configs]
    colors       = [c["color"] for c in configs]

    x = np.arange(len(labels))
    bar_w = 0.55

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle("Figure 4 — LUMI configuration comparison (8h wall, QuixBugs 40 tasks)",
                 fontsize=12, y=1.01)

    # Panel A — tasks completed
    ax = axes[0]
    bars = ax.bar(x, tasks_done, color=colors, width=bar_w, edgecolor="white", zorder=3)
    ax.axhline(40, color="#555", linewidth=1, linestyle=":", zorder=2)
    ax.text(2.35, 40.5, "all 40", fontsize=8, color="#555")
    for bar, val in zip(bars, tasks_done):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Tasks completed out of 40")
    ax.set_ylim(0, 46)
    ax.set_title("Tasks completed", fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Panel B — solve rate
    ax = axes[1]
    bars = ax.bar(x, pass_rates, color=colors, width=bar_w, edgecolor="white", zorder=3)
    for bar, val in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Solve rate (% of completed tasks)")
    ax.set_ylim(0, 58)
    ax.set_title("Solve rate (PASS %)", fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Panel C — model load time
    ax = axes[2]
    bars = ax.bar(x, model_loads, color=colors, width=bar_w, edgecolor="white", zorder=3)
    for bar, val in zip(bars, model_loads):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f} min", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Model load time (minutes)")
    ax.set_ylim(0, 46)
    ax.set_title("Model load time", fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.annotate("* likely Lustre\ncache effect",
                xy=(2, 4.2), xytext=(1.5, 18),
                arrowprops=dict(arrowstyle="->", color="#555"),
                fontsize=8, color="#555")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig4_config_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Generating figures...")
    fig1_phase_pie()
    fig2_phase_bars()
    fig3_inference_vs_step()
    fig4_config_comparison()
    print("Done. Figures saved to", OUT_DIR)
