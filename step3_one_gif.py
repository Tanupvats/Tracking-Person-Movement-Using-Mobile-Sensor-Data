
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from src.visualize import animate_tracking
from src.scenarios import interpolate_truth


def main():
    name = sys.argv[1]
    with open(f"cache/{name}.pkl", "rb") as fh:
        r = pickle.load(fh)

    # Interpolate truth onto filter times for per-step error
    truth_at_t = np.stack(
        [np.interp(r["times"], r["truth_t"], r["truth_pos"][:, 0]),
         np.interp(r["times"], r["truth_t"], r["truth_pos"][:, 1])],
        axis=1,
    )

    out = f"outputs/{name}_track.gif"
    animate_tracking(
        times=r["times"],
        truth_xy=truth_at_t,
        gps_xy=r["gps_xy"],
        gps_times=r["gps_t"],
        est_xy=r["poses"][:, :2],
        est_cov=r["covs"],
        mode_probs=r["modes"],
        out_path=out,
        title=r["title"],
        n_frames=80,
        fps=16,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
