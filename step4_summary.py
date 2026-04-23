
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.visualize import THEME, _apply_theme


def load_all():
    results = {}
    for fn in sorted(os.listdir("cache")):
        if fn == "summary.pkl" or not fn.endswith(".pkl"):
            continue
        name = fn[:-4]
        with open(f"cache/{fn}", "rb") as fh:
            results[name] = pickle.load(fh)
    return results


def plot_summary(results, out_path):
    _apply_theme()
    names = list(results.keys())
    gps_rmse = [results[n]["gps_rmse"] for n in names]
    est_rmse = [results[n]["est_rmse"] for n in names]
    sm_rmse  = [results[n]["sm_rmse"]  for n in names]

    x = np.arange(len(names))
    w = 0.26

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w, gps_rmse, w, label="Raw GPS", color=THEME["gps"])
    ax.bar(x,     est_rmse, w, label="INS/GNSS fused (online)", color=THEME["est"])
    ax.bar(x + w, sm_rmse,  w, label="RTS smoothed (offline)", color=THEME["ctrv"])
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", " ") for n in names], rotation=10)
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Position RMSE per scenario — INS/GNSS fusion vs. raw GPS")
    ax.grid(True, axis="y")
    ax.legend(framealpha=0.85, loc="upper left")

    for i, (g, e) in enumerate(zip(gps_rmse, est_rmse)):
        imp = (1 - e / max(g, 1e-6)) * 100
        ax.text(i, max(g, e) + 0.2, f"−{imp:.0f}%",
                color=THEME["text"], ha="center", fontsize=10,
                fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def write_markdown(results, out_path):
    lines = [
        "# Benchmark Results\n",
        "| Scenario | Samples | Raw-GPS RMSE | Fused RMSE | Smoothed RMSE | "
        "CEP50 | Max err | Improvement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    total_imp = []
    for name, r in results.items():
        imp = (1 - r["est_rmse"] / max(r["gps_rmse"], 1e-6)) * 100
        total_imp.append(imp)
        lines.append(
            f"| {name.replace('_', ' ')} "
            f"| {r['num_samples']} "
            f"| {r['gps_rmse']:.2f} m "
            f"| {r['est_rmse']:.2f} m "
            f"| {r['sm_rmse']:.2f} m "
            f"| {r['est_cep50']:.2f} m "
            f"| {r['est_max']:.2f} m "
            f"| **{imp:.1f} %** |"
        )
    avg = sum(total_imp) / len(total_imp)
    lines.append(f"\n**Mean improvement over raw GPS: {avg:.1f} %**")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))


def main():
    results = load_all()
    plot_summary(results, "outputs/summary.png")
    print("wrote outputs/summary.png")
    write_markdown(results, "outputs/summary.md")
    print("wrote outputs/summary.md")


if __name__ == "__main__":
    main()
