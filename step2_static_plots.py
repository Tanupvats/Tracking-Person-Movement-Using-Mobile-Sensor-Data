
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.visualize import TrajectoryPlot, plot_trajectories


def main():
    for fn in sorted(os.listdir("cache")):
        if fn == "summary.pkl" or not fn.endswith(".pkl"):
            continue
        name = fn[:-4]
        with open(f"cache/{fn}", "rb") as fh:
            r = pickle.load(fh)

        metrics = {
            "Raw GPS RMSE":  f"{r['gps_rmse']:.2f} m",
            "Raw GPS CEP50": f"{r['gps_cep50']:.2f} m",
            "Fused RMSE":    f"{r['est_rmse']:.2f} m",
            "Fused CEP50":   f"{r['est_cep50']:.2f} m",
            "Fused max err": f"{r['est_max']:.2f} m",
            "Smoothed RMSE": f"{r['sm_rmse']:.2f} m",
            "Improvement":
                f"{(1 - r['est_rmse'] / max(r['gps_rmse'], 1e-6)) * 100:.1f} %",
        }
        tp = TrajectoryPlot(
            title=r["title"],
            truth_xy=r["truth_pos"],
            gps_xy=r["gps_xy"],
            est_xy=r["poses"][:, :2],
            est_cov=r["covs"],
            metrics=metrics,
        )
        out = f"outputs/{name}_traj.png"
        plot_trajectories(tp, out)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
