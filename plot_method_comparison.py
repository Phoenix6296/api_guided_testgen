import argparse
import csv
import json
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LIB_ORDER = ["tf", "torch", "sklearn", "xgb", "jax"]
DEFAULT_METHOD_ORDER = [
    "zero_shot",
    "similarity",
    "diversity",
    "hybrid",
    "basic_rag_all",
    "basic_rag_apidoc",
    "basic_rag_sos",
    "basic_rag_issues",
    "basic_rag_repos",
    "api_rag_all",
    "api_rag_apidoc",
    "api_rag_sos",
    "api_rag_issues",
    "api_rag_repos",
]


def parse_eval_log(log_path):
    eval_header = re.compile(r"^=== EVALUATE\s+(\w+)\s+([\w_]+)\s+===")
    metric_line = re.compile(r"^(parse|exec|pass) rate:\s+.+?=\s+(-?\d+(?:\.\d+)?)")

    metrics = {}
    current = None

    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            header_match = eval_header.match(line)
            if header_match:
                lib, method = header_match.groups()
                current = (lib, method)
                metrics.setdefault(current, {})
                continue

            if not current:
                continue

            metric_match = metric_line.match(line)
            if metric_match:
                metric_name, value = metric_match.groups()
                metrics[current][f"{metric_name}_rate"] = float(value)

    return metrics


def parse_coverage_dir(coverage_dir):
    coverage = {}

    for file_name in sorted(os.listdir(coverage_dir)):
        if not file_name.endswith(".json"):
            continue

        stem = file_name[:-5]
        if "_" not in stem:
            continue

        lib, method = stem.split("_", 1)
        file_path = os.path.join(coverage_dir, file_name)
        try:
            with open(file_path, encoding="utf-8") as f:
                payload = json.load(f)
            percent = payload.get("totals", {}).get("percent_covered")
            if percent is not None:
                coverage[(lib, method)] = float(percent)
        except (json.JSONDecodeError, OSError, ValueError):
            continue

    return coverage


def build_rows(eval_metrics, coverage_metrics):
    all_keys = sorted(set(eval_metrics.keys()) | set(coverage_metrics.keys()))
    rows = []

    for lib, method in all_keys:
        row = {
            "library": lib,
            "method": method,
            "parse_rate": None,
            "exec_rate": None,
            "pass_rate": None,
            "coverage": None,
        }
        row.update(eval_metrics.get((lib, method), {}))
        if (lib, method) in coverage_metrics:
            row["coverage"] = coverage_metrics[(lib, method)]
        rows.append(row)

    return rows


def ordered_unique(values, preferred_order):
    seen = set(values)
    ordered = [v for v in preferred_order if v in seen]
    leftovers = sorted(v for v in seen if v not in preferred_order)
    return ordered + leftovers


def build_matrix(rows, libs, methods, metric):
    lookup = {(r["library"], r["method"]): r.get(metric) for r in rows}
    return [[lookup.get((lib, method), math.nan) for method in methods] for lib in libs]


def plot_heatmap(rows, libs, methods, metric, title, out_path):
    data = np.array(build_matrix(rows, libs, methods, metric), dtype=float)

    fig_w = max(12, len(methods) * 0.85)
    fig_h = max(4, len(libs) * 0.8 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_yticks(range(len(libs)))
    ax.set_yticklabels(libs)
    ax.set_title(title)

    for i, lib in enumerate(libs):
        for j, method in enumerate(methods):
            value = data[i][j]
            if not math.isnan(float(value)):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Percent")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_method_average_bars(rows, methods, out_path):
    method_to_pass = {m: [] for m in methods}
    method_to_cov = {m: [] for m in methods}

    for row in rows:
        method = row["method"]
        if method not in method_to_pass:
            continue
        if row.get("pass_rate") is not None:
            method_to_pass[method].append(row["pass_rate"])
        if row.get("coverage") is not None:
            method_to_cov[method].append(row["coverage"])

    pass_avg = [sum(method_to_pass[m]) / len(method_to_pass[m]) if method_to_pass[m] else math.nan for m in methods]
    cov_avg = [sum(method_to_cov[m]) / len(method_to_cov[m]) if method_to_cov[m] else math.nan for m in methods]

    x = list(range(len(methods)))
    width = 0.38

    fig_w = max(12, len(methods) * 0.85)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    ax.bar([i - width / 2 for i in x], pass_avg, width=width, label="Avg pass rate")
    ax.bar([i + width / 2 for i in x], cov_avg, width=width, label="Avg line coverage")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent")
    ax.set_title("Average performance across libraries")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_pass_vs_coverage(rows, out_path):
    libs = sorted({r["library"] for r in rows})
    cmap = plt.get_cmap("tab10")
    colors = {lib: cmap(i % 10) for i, lib in enumerate(libs)}

    fig, ax = plt.subplots(figsize=(8, 6))

    for lib in libs:
        xs = []
        ys = []
        for row in rows:
            if row["library"] != lib:
                continue
            if row.get("coverage") is None or row.get("pass_rate") is None:
                continue
            xs.append(row["coverage"])
            ys.append(row["pass_rate"])
        if xs:
            ax.scatter(xs, ys, s=55, alpha=0.8, color=colors[lib], label=lib)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Line coverage (%)")
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("Pass rate vs line coverage")
    ax.grid(alpha=0.3)
    ax.legend(title="Library")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_summary_csv(rows, out_path):
    headers = ["library", "method", "parse_rate", "exec_rate", "pass_rate", "coverage"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Create method-comparison plots from run logs and coverage JSON outputs.")
    parser.add_argument("--iter", default="qwen2.5-coder_7b-instruct", dest="iter_name", help="Run iteration name under out/ and log/.")
    parser.add_argument("--log", default=None, help="Path to full run log. Defaults to log/<iter>/full_alllibs_allapis.log")
    parser.add_argument("--coverage-dir", default=None, help="Path to coverage JSON directory. Defaults to out/<iter>/coverage")
    parser.add_argument("--plots-dir", default=None, help="Output directory for plots/CSV. Defaults to out/<iter>/plots")
    args = parser.parse_args()

    log_path = args.log or os.path.join("log", args.iter_name, "full_alllibs_allapis.log")
    coverage_dir = args.coverage_dir or os.path.join("out", args.iter_name, "coverage")
    plots_dir = args.plots_dir or os.path.join("out", args.iter_name, "plots")

    os.makedirs(plots_dir, exist_ok=True)

    eval_metrics = parse_eval_log(log_path)
    coverage_metrics = parse_coverage_dir(coverage_dir)
    rows = build_rows(eval_metrics, coverage_metrics)

    if not rows:
        raise ValueError("No rows parsed. Check the log and coverage paths.")

    libs = ordered_unique([r["library"] for r in rows], DEFAULT_LIB_ORDER)
    methods = ordered_unique([r["method"] for r in rows], DEFAULT_METHOD_ORDER)

    write_summary_csv(rows, os.path.join(plots_dir, "method_comparison_summary.csv"))

    plot_heatmap(
        rows,
        libs,
        methods,
        metric="pass_rate",
        title="Pass rate by library and method",
        out_path=os.path.join(plots_dir, "method_pass_rate_heatmap.png"),
    )
    plot_heatmap(
        rows,
        libs,
        methods,
        metric="coverage",
        title="Line coverage by library and method",
        out_path=os.path.join(plots_dir, "method_coverage_heatmap.png"),
    )
    plot_heatmap(
        rows,
        libs,
        methods,
        metric="exec_rate",
        title="Execution rate by library and method",
        out_path=os.path.join(plots_dir, "method_exec_rate_heatmap.png"),
    )
    plot_method_average_bars(rows, methods, os.path.join(plots_dir, "method_average_pass_vs_coverage.png"))
    plot_pass_vs_coverage(rows, os.path.join(plots_dir, "pass_vs_coverage_scatter.png"))

    print(f"Wrote comparison CSV and plots to {plots_dir}")


if __name__ == "__main__":
    main()
