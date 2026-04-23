

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import numpy as np


THEME = {
    "bg":        "#0e1117",
    "panel":     "#161a22",
    "grid":      "#2a2f3a",
    "text":      "#e5e7eb",
    "truth":     "#60a5fa",   # blue
    "gps":       "#f97316",   # orange
    "est":       "#10b981",   # green
    "ellipse":   "#10b981",
    "cv":        "#60a5fa",
    "ctrv":      "#f472b6",
    "ca":        "#fbbf24",
    "muted":     "#9ca3af",
}

# Labels/colors for mode stackplot (straight-cruise vs manoeuvre)
MODE_LABELS = ["Cruise", "Manoeuvre"]
MODE_COLORS = [THEME["cv"], THEME["ctrv"]]


def _apply_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor":   THEME["bg"],
        "axes.facecolor":     THEME["panel"],
        "axes.edgecolor":     THEME["grid"],
        "axes.labelcolor":    THEME["text"],
        "axes.titlecolor":    THEME["text"],
        "text.color":         THEME["text"],
        "xtick.color":        THEME["text"],
        "ytick.color":        THEME["text"],
        "grid.color":         THEME["grid"],
        "grid.alpha":         0.4,
        "grid.linestyle":     "--",
        "font.size":          10,
        "axes.titlesize":     12,
        "axes.titleweight":   "bold",
        "legend.facecolor":   THEME["panel"],
        "legend.edgecolor":   THEME["grid"],
        "legend.labelcolor":  THEME["text"],
        "savefig.facecolor":  THEME["bg"],
    })





def cov_ellipse(cov: np.ndarray, n_std: float = 2.0, **kwargs) -> Ellipse:
    cov2 = cov[:2, :2]
    vals, vecs = np.linalg.eigh(cov2)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.clip(vals, 1e-9, None)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    return Ellipse(xy=(0, 0), width=width, height=height, angle=theta, **kwargs)




@dataclass
class TrajectoryPlot:
    title: str
    truth_xy: np.ndarray
    gps_xy: np.ndarray
    est_xy: np.ndarray
    est_cov: Optional[np.ndarray] = None
    metrics: Optional[dict] = None


def plot_trajectories(tp: TrajectoryPlot, out_path: str) -> None:
    _apply_theme()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(tp.truth_xy[:, 0], tp.truth_xy[:, 1],
            color=THEME["truth"], lw=2.5, label="Ground truth", alpha=0.9, zorder=3)
    ax.scatter(tp.gps_xy[:, 0], tp.gps_xy[:, 1],
               color=THEME["gps"], s=14, alpha=0.55, label="Raw GPS", zorder=2)
    ax.plot(tp.est_xy[:, 0], tp.est_xy[:, 1],
            color=THEME["est"], lw=2.0, linestyle="--",
            label="Fused estimate (INS/GNSS)", zorder=4)

    if tp.est_cov is not None and len(tp.est_cov) > 0:
        # Sparse uncertainty ellipses along the track
        N = len(tp.est_xy)
        step = max(1, N // 30)
        for i in range(0, N, step):
            e = cov_ellipse(tp.est_cov[i], n_std=2.0,
                            edgecolor=THEME["ellipse"], facecolor="none",
                            lw=0.8, alpha=0.45)
            e.set_center(tp.est_xy[i])
            ax.add_patch(e)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(tp.title)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", framealpha=0.85)

    if tp.metrics:
        txt = "\n".join(f"{k}: {v}" for k, v in tp.metrics.items())
        ax.text(
            0.02, 0.98, txt, transform=ax.transAxes,
            fontsize=9, color=THEME["text"], va="top", ha="left",
            family="monospace",
            bbox=dict(facecolor=THEME["panel"], edgecolor=THEME["grid"], alpha=0.9, pad=6),
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)




def animate_tracking(
    times: np.ndarray,
    truth_xy: np.ndarray,
    gps_xy: np.ndarray,
    gps_times: np.ndarray,
    est_xy: np.ndarray,
    est_cov: np.ndarray,
    mode_probs: Optional[np.ndarray],
    out_path: str,
    title: str,
    n_frames: int = 120,
    fps: int = 20,
) -> None:
    """
    Render a dashboard animation:
        * Main panel (left): bird-s-eye view with trail, uncertainty, GPS dots
        * Top-right: radial error vs. time
        * Bottom-right: IMM mode probabilities stacked area
    """
    _apply_theme()

    # Down-sample to n_frames
    N = len(times)
    idx = np.linspace(0, N - 1, n_frames).astype(int)

    # Compute per-frame GPS subset (those with t <= times[idx[i]])
    # Precompute error signal (truth vs estimate at filter timestamps)
    err = np.linalg.norm(est_xy - truth_xy, axis=1)

    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1.0], height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.22)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_err = fig.add_subplot(gs[0, 1])
    ax_imm = fig.add_subplot(gs[1, 1])

    # ---- Map axis --------------------------------------------------------
    # Establish bounds using all of truth + estimate
    all_xy = np.vstack([truth_xy, est_xy, gps_xy])
    xmin, xmax = all_xy[:, 0].min(), all_xy[:, 0].max()
    ymin, ymax = all_xy[:, 1].min(), all_xy[:, 1].max()
    pad = max(5.0, 0.1 * max(xmax - xmin, ymax - ymin))
    ax_map.set_xlim(xmin - pad, xmax + pad)
    ax_map.set_ylim(ymin - pad, ymax + pad)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.set_title(title)
    ax_map.grid(True)

    (ln_truth,)  = ax_map.plot([], [], color=THEME["truth"], lw=2.2, label="Ground truth", zorder=3)
    sc_gps       = ax_map.scatter([], [], color=THEME["gps"], s=18, alpha=0.55, label="GPS", zorder=2)
    (ln_est,)    = ax_map.plot([], [], color=THEME["est"], lw=2.0, linestyle="--", label="Fused (INS/GNSS)", zorder=5)
    pt_est,      = ax_map.plot([], [], "o", color=THEME["est"], ms=9, zorder=6,
                               markeredgecolor="white", markeredgewidth=1.2)
    # Uncertainty ellipse (mutable)
    ell = cov_ellipse(np.eye(4), n_std=2.0, edgecolor=THEME["ellipse"],
                      facecolor=THEME["ellipse"], alpha=0.18, lw=1.5)
    ax_map.add_patch(ell)
    ax_map.legend(loc="upper left", framealpha=0.9)

    # ---- Error axis ------------------------------------------------------
    ax_err.set_title("Radial error  |estimate − truth|  (m)")
    ax_err.set_xlim(times[0], times[-1])
    ax_err.set_ylim(0, max(err.max() * 1.15, 1.0))
    ax_err.set_xlabel("Time (s)")
    ax_err.grid(True)
    (ln_err,) = ax_err.plot([], [], color=THEME["est"], lw=1.6)
    ax_err.fill_between([], [], [], color=THEME["est"], alpha=0.2)
    err_fill = [None]  # placeholder for PolyCollection we regenerate each frame

    # ---- IMM axis --------------------------------------------------------
    ax_imm.set_title("Adaptive mode (cruise ↔ manoeuvre)")
    ax_imm.set_xlim(times[0], times[-1])
    ax_imm.set_ylim(0, 1)
    ax_imm.set_xlabel("Time (s)")
    ax_imm.set_ylabel("p(model)")
    ax_imm.grid(True)

    has_modes = mode_probs is not None and len(mode_probs) > 0
    if has_modes:
        # mode_probs has one entry per update; interpolate onto times
        m_t = np.linspace(times[0], times[-1], len(mode_probs))
        mode_interp = np.stack(
            [np.interp(times, m_t, mode_probs[:, k]) for k in range(mode_probs.shape[1])],
            axis=1,
        )
        labels = ["CV", "CTRV", "CA"]
        colors = [THEME["cv"], THEME["ctrv"], THEME["ca"]]
        # Placeholder for stackplot; we re-create it inside update
    else:
        mode_interp = None
        ax_imm.text(0.5, 0.5, "(no IMM data)", transform=ax_imm.transAxes,
                    ha="center", va="center", color=THEME["muted"])

    time_text = fig.text(0.012, 0.975, "", color=THEME["text"],
                         family="monospace", fontsize=10, va="top")

    def init():
        ln_truth.set_data([], [])
        ln_est.set_data([], [])
        pt_est.set_data([], [])
        sc_gps.set_offsets(np.empty((0, 2)))
        ln_err.set_data([], [])
        ell.set_center((0, 0))
        return (ln_truth, ln_est, pt_est, sc_gps, ln_err, ell)

    def update(i):
        k = idx[i]
        # Map
        ln_truth.set_data(truth_xy[:k + 1, 0], truth_xy[:k + 1, 1])
        ln_est.set_data(est_xy[:k + 1, 0], est_xy[:k + 1, 1])
        pt_est.set_data([est_xy[k, 0]], [est_xy[k, 1]])

        gps_mask = gps_times <= times[k]
        sc_gps.set_offsets(gps_xy[gps_mask]) if gps_mask.any() else sc_gps.set_offsets(np.empty((0, 2)))

        # Covariance ellipse
        new_ell = cov_ellipse(est_cov[k], n_std=2.0)
        ell.set_width(new_ell.width)
        ell.set_height(new_ell.height)
        ell.set_angle(new_ell.angle)
        ell.set_center(est_xy[k])

        # Error trace
        ln_err.set_data(times[:k + 1], err[:k + 1])
        # Refresh fill (PolyCollection remove loop, not ArtistList.clear)
        for coll in list(ax_err.collections):
            coll.remove()
        ax_err.fill_between(times[:k + 1], 0, err[:k + 1], color=THEME["est"], alpha=0.25)

        # IMM stack plot
        if has_modes:
            for coll in list(ax_imm.collections):
                coll.remove()
            ax_imm.stackplot(
                times[:k + 1],
                *[mode_interp[:k + 1, c] for c in range(mode_interp.shape[1])],
                labels=MODE_LABELS[: mode_interp.shape[1]],
                colors=MODE_COLORS[: mode_interp.shape[1]],
                alpha=0.85,
            )
            if i == 1:
                ax_imm.legend(loc="upper right", framealpha=0.9, fontsize=8)

        time_text.set_text(f"t = {times[k]:6.2f} s    err = {err[k]:5.2f} m")
        return (ln_truth, ln_est, pt_est, sc_gps, ln_err, ell)

    ani = animation.FuncAnimation(
        fig, update, init_func=init, frames=len(idx),
        blit=False, interval=1000 / fps,
    )
    # Use pillow writer for GIF
    writer = animation.PillowWriter(fps=fps)
    ani.save(out_path, writer=writer, dpi=100)
    plt.close(fig)
