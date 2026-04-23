
import os, sys, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from src.scenarios import (
    SensorConfig, cep50, extract_gps_only, interpolate_truth, max_err, rmse,
    stream_sensors,
    traj_circle, traj_figure_eight, traj_highway, traj_pedestrian_mixed,
    traj_urban_canyon,
)
from src.ins_tracker import INSTracker, rts_smooth_track


SCENARIOS = {
    "circle": {
        "title": "Circular motion (r=20m, v=2m/s)",
        "factory": lambda: traj_circle(T=40.0, radius=20.0, speed=2.0),
        "sensor_cfg": SensorConfig(gps_noise=2.5, gps_multipath_prob=0.02,
                                   gps_multipath_magnitude=15.0,
                                   acc_noise=0.18, gyro_noise=0.008, seed=1),
    },
    "figure_eight": {
        "title": "Figure-eight (A=15m, v≈1.5m/s)",
        "factory": lambda: traj_figure_eight(T=60.0, A=15.0, speed=1.5),
        "sensor_cfg": SensorConfig(gps_noise=3.0, gps_multipath_prob=0.03,
                                   gps_multipath_magnitude=18.0,
                                   acc_noise=0.2, gyro_noise=0.01, seed=2),
    },
    "urban_canyon": {
        "title": "Urban canyon (GPS outage 35-55s + heavy multipath)",
        "factory": lambda: traj_urban_canyon(T=90.0),
        "sensor_cfg": SensorConfig(gps_noise=4.0,
                                   gps_outage_intervals=[(35.0, 55.0)],
                                   gps_multipath_prob=0.15,
                                   gps_multipath_magnitude=25.0,
                                   acc_noise=0.25, gyro_noise=0.012, seed=3),
    },
    "highway": {
        "title": "Vehicle highway (accel → cruise → lane change → brake)",
        "factory": lambda: traj_highway(T=55.0),
        "sensor_cfg": SensorConfig(gps_noise=3.0,
                                   acc_noise=0.22, gyro_noise=0.009, seed=4),
    },
    "pedestrian": {
        "title": "Pedestrian: walk → stop → jog (ZUPT + adaptive Q)",
        "factory": lambda: traj_pedestrian_mixed(T=85.0),
        "sensor_cfg": SensorConfig(gps_noise=3.5,
                                   gps_multipath_prob=0.06,
                                   gps_multipath_magnitude=20.0,
                                   acc_noise=0.22, gyro_noise=0.01, seed=5),
    },
}


def run_one(name, cfg):
    print(f"  {name}...", flush=True)
    truth = cfg["factory"]()
    samples = list(stream_sensors(truth, cfg["sensor_cfg"]))

    tr = INSTracker()
    tr.initialize(pos=truth.pos[0], heading=truth.heading[0],
                  speed=float(np.hypot(*truth.vel[0])), t0=truth.t[0])
    t0 = time.time()
    for s in samples:
        tr.step(s)
    wall = time.time() - t0

    times, poses, covs = tr.trajectory()
    poses_sm, covs_sm = rts_smooth_track(times, poses, covs)
    modes = tr.mode_history()

    gt, gps_xy = extract_gps_only(samples)
    truth_at_t = interpolate_truth(times, truth)
    truth_at_gps = interpolate_truth(gt, truth)

    result = {
        "title": cfg["title"],
        "truth_t": truth.t,
        "truth_pos": truth.pos,
        "times": times,
        "poses": poses,
        "covs": covs,
        "poses_sm": poses_sm,
        "covs_sm": covs_sm,
        "modes": modes,
        "gps_t": gt,
        "gps_xy": gps_xy,
        "wall_time": wall,
        "num_samples": len(samples),
        # metrics
        "gps_rmse": rmse(gps_xy, truth_at_gps),
        "gps_cep50": cep50(gps_xy, truth_at_gps),
        "gps_max": max_err(gps_xy, truth_at_gps),
        "est_rmse": rmse(poses[:, :2], truth_at_t),
        "est_cep50": cep50(poses[:, :2], truth_at_t),
        "est_max": max_err(poses[:, :2], truth_at_t),
        "sm_rmse": rmse(poses_sm[:, :2], truth_at_t),
        "sm_cep50": cep50(poses_sm[:, :2], truth_at_t),
    }
    return result


def main():
    os.makedirs("cache", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    print("Running all scenarios...")
    results = {}
    for name, cfg in SCENARIOS.items():
        res = run_one(name, cfg)
        with open(f"cache/{name}.pkl", "wb") as fh:
            pickle.dump(res, fh)
        results[name] = res
        imp = (1 - res["est_rmse"] / max(res["gps_rmse"], 1e-6)) * 100
        print(f"    GPS RMSE={res['gps_rmse']:.2f}  Fused={res['est_rmse']:.2f}  "
              f"Smoothed={res['sm_rmse']:.2f}  improvement={imp:.1f}%  "
              f"({res['num_samples']} samples in {res['wall_time']:.2f}s)")

    # Save summary
    with open("cache/summary.pkl", "wb") as fh:
        pickle.dump(
            {k: {kk: v for kk, v in r.items()
                 if not isinstance(v, np.ndarray)}
             for k, r in results.items()},
            fh,
        )
    print("Step 1 done. All results cached to cache/*.pkl.")


if __name__ == "__main__":
    main()
