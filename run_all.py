

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.scenarios import (
    SensorConfig, cep50, extract_gps_only, interpolate_truth, max_err, rmse,
    stream_sensors,
    traj_circle, traj_figure_eight, traj_highway, traj_pedestrian_mixed,
    traj_urban_canyon,
)
from src.ins_tracker import INSTracker, rts_smooth_track
from src.visualize import TrajectoryPlot, animate_tracking, plot_trajectories



SCENARIOS = {
    "circle": {
        "title": "Circular motion (r=20m, v=2m/s)",
        "factory": lambda: traj_circle(T=40.0, radius=20.0, speed=2.0),
        "sensor_cfg": SensorConfig(
            gps_noise=2.5, gps_multipath_prob=0.02, gps_multipath_magnitude=15.0,
            acc_noise=0.18, gyro_noise=0.008, seed=1,
        ),
    },
    "figure_eight": {
        "title": "Figure-eight (A=15m, v≈1.5m/s)",
        "factory": lambda: traj_figure_eight(T=60.0, A=15.0, speed=1.5),
        "sensor_cfg": SensorConfig(
            gps_noise=3.0, gps_multipath_prob=0.03, gps_multipath_magnitude=18.0,
            acc_noise=0.2, gyro_noise=0.01, seed=2,
        ),
    },
    "urban_canyon": {
        "title": "Urban canyon walk (GPS outage 35-55s, high multipath)",
        "factory": lambda: traj_urban_canyon(T=90.0),
        "sensor_cfg": SensorConfig(
            gps_noise=4.0,
            gps_outage_intervals=[(35.0, 55.0)],
            gps_multipath_prob=0.15, gps_multipath_magnitude=25.0,
            acc_noise=0.25, gyro_noise=0.012, seed=3,
        ),
    },
    "highway": {
        "title": "Vehicle highway (accel + lane change + brake)",
        "factory": lambda: traj_highway(T=55.0),
        "sensor_cfg": SensorConfig(
            gps_noise=3.0, gps_multipath_prob=0.0,
            acc_noise=0.22, gyro_noise=0.009, seed=4,
        ),
    },
    "pedestrian": {
        "title": "Pedestrian: walk → stop → jog (tests ZUPT + IMM switching)",
        "factory": lambda: traj_pedestrian_mixed(T=85.0),
        "sensor_cfg": SensorConfig(
            gps_noise=3.5, gps_multipath_prob=0.06, gps_multipath_magnitude=20.0,
            acc_noise=0.22, gyro_noise=0.01, seed=5,
        ),
    },
}



def run_scenario(name: str, cfg: dict, out_dir: str) -> dict:
    print(f"\n=== Scenario: {name} ===")
    truth = cfg["factory"]()
    sensor_cfg = cfg["sensor_cfg"]
    samples = list(stream_sensors(truth, sensor_cfg))

    # Initialise tracker from truth (one-shot initial fix)
    tracker = INSTracker()
    tracker.initialize(
        pos=truth.pos[0],
        heading=truth.heading[0],
        speed=float(np.hypot(*truth.vel[0])),
        t0=truth.t[0],
    )

    t_start = time.time()
    for s in samples:
        tracker.step(s)
    wall = time.time() - t_start
    print(f"    processed {len(samples):6d} samples in {wall:.2f} s  "
          f"({len(samples)/wall:.0f} Hz)")

    times, poses, covs = tracker.trajectory()
    # Offline smoothing (RTS) over the pose track
    poses_smoothed, covs_smoothed = rts_smooth_track(times, poses, covs)
    mode_probs = tracker.mode_history()

    # --- Metrics vs. truth ---
    truth_at_times = interpolate_truth(times, truth)
    est_xy = poses[:, :2]
    est_xy_smooth = poses_smoothed[:, :2]
    e_rmse = rmse(est_xy, truth_at_times)
    e_cep = cep50(est_xy, truth_at_times)
    e_max = max_err(est_xy, truth_at_times)
    s_rmse = rmse(est_xy_smooth, truth_at_times)
    s_cep = cep50(est_xy_smooth, truth_at_times)

    # Raw GPS baseline
    gps_t, gps_xy = extract_gps_only(samples)
    gps_truth = interpolate_truth(gps_t, truth)
    g_rmse = rmse(gps_xy, gps_truth)
    g_cep  = cep50(gps_xy, gps_truth)
    g_max  = max_err(gps_xy, gps_truth)

    metrics = {
        "Fused RMSE":        f"{e_rmse:.2f} m",
        "Fused CEP50":       f"{e_cep:.2f} m",
        "Fused max err":     f"{e_max:.2f} m",
        "Smoothed RMSE":     f"{s_rmse:.2f} m",
        "Smoothed CEP50":    f"{s_cep:.2f} m",
        "Raw GPS RMSE":      f"{g_rmse:.2f} m",
        "Raw GPS CEP50":     f"{g_cep:.2f} m",
        "Raw GPS max err":   f"{g_max:.2f} m",
        "Improvement":       f"{(1 - e_rmse / max(g_rmse, 1e-6)) * 100:.1f} %",
    }
    for k, v in metrics.items():
        print(f"    {k:22s} {v}")

    # --- Static plot ---
    tp = TrajectoryPlot(
        title=cfg["title"],
        truth_xy=truth.pos,
        gps_xy=gps_xy,
        est_xy=est_xy,
        est_cov=covs,
        metrics=metrics,
    )
    plot_trajectories(tp, os.path.join(out_dir, f"{name}_traj.png"))

    # --- Dashboard GIF ---
    animate_tracking(
        times=times,
        truth_xy=truth_at_times,
        gps_xy=gps_xy,
        gps_times=gps_t,
        est_xy=est_xy,
        est_cov=covs,
        mode_probs=mode_probs,
        out_path=os.path.join(out_dir, f"{name}_track.gif"),
        title=cfg["title"],
        n_frames=110,
        fps=18,
    )

    # Strip 'm', '%' for numeric summary
    numeric = {
        "est_rmse":  e_rmse,
        "est_cep50": e_cep,
        "est_max":   e_max,
        "smooth_rmse": s_rmse,
        "smooth_cep50": s_cep,
        "gps_rmse":  g_rmse,
        "gps_cep50": g_cep,
        "gps_max":   g_max,
    }
    return numeric




def plot_summary(results: dict, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.visualize import THEME, _apply_theme
    _apply_theme()

    names = list(results.keys())
    gps_rmse = [results[n]["gps_rmse"] for n in names]
    est_rmse = [results[n]["est_rmse"] for n in names]
    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w / 2, gps_rmse, w, label="Raw GPS", color=THEME["gps"])
    ax.bar(x + w / 2, est_rmse, w, label="IMM-UKF fused", color=THEME["est"])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Fused-tracker vs raw-GPS position RMSE per scenario")
    ax.grid(True, axis="y")
    ax.legend(framealpha=0.85)

    for i, (g, e) in enumerate(zip(gps_rmse, est_rmse)):
        imp = (1 - e / max(g, 1e-6)) * 100
        ax.text(i, max(g, e) + 0.15, f"−{imp:.0f}%",
                color=THEME["text"], ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def write_summary_md(results: dict, out_path: str) -> None:
    lines = ["# Benchmark Summary\n"]
    lines.append(
        "| Scenario | Raw-GPS RMSE (m) | Fused RMSE (m) | CEP50 (m) | Max err (m) | Improvement |\n"
        "|---|---:|---:|---:|---:|---:|"
    )
    for name, r in results.items():
        imp = (1 - r["est_rmse"] / max(r["gps_rmse"], 1e-6)) * 100
        lines.append(
            f"| {name} | {r['gps_rmse']:.2f} | {r['est_rmse']:.2f} "
            f"| {r['est_cep50']:.2f} | {r['est_max']:.2f} | {imp:.1f}% |"
        )
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))




def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for name, cfg in SCENARIOS.items():
        try:
            results[name] = run_scenario(name, cfg, out_dir)
        except Exception as exc:  # pragma: no cover
            print(f"    !! scenario {name} failed: {exc}")
            import traceback; traceback.print_exc()

    plot_summary(results, os.path.join(out_dir, "summary.png"))
    write_summary_md(results, os.path.join(out_dir, "summary.md"))

    print("\nAll done. Outputs in", out_dir)


if __name__ == "__main__":
    main()
