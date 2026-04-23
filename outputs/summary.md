# Benchmark Results

| Scenario | Samples | Raw-GPS RMSE | Fused RMSE | Smoothed RMSE | CEP50 | Max err | Improvement |
|---|---:|---:|---:|---:|---:|---:|---:|
| circle | 4439 | 3.31 m | 1.87 m | 1.87 m | 1.58 m | 3.92 m | **43.4 %** |
| figure eight | 6659 | 6.88 m | 2.60 m | 2.60 m | 1.68 m | 6.66 m | **62.2 %** |
| highway | 6104 | 4.23 m | 1.85 m | 1.85 m | 1.47 m | 4.34 m | **56.3 %** |
| pedestrian | 9434 | 8.18 m | 1.80 m | 1.80 m | 1.51 m | 4.54 m | **78.0 %** |
| urban canyon | 9968 | 8.41 m | 5.81 m | 5.81 m | 3.24 m | 17.26 m | **31.0 %** |

**Mean improvement over raw GPS: 54.2 %**