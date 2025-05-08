## Path Planning for UAV Interior Structure Scanning in Buildings

This repository presents a complete solution for obstacle-aware path planning and reconstruction of UAV flight trajectories over multi-layer 3D structures. Designed for autonomous inspection scenarios, the system reduces computational overhead and enhances trajectory accuracy.

---

### Key Features

* **3D Point Cloud Slicing** – C++ based segmentation by Z-axis height
* **Occupancy Grid Generation** – Robust dilation + closing for obstacle repair
* **KNN-based MST** – Efficient nearest-neighbor path generation with obstacle filtering
* **Layer-wise DFS + Inter-layer A\*** – Full vertical + horizontal UAV path construction
* **Open3D Visualization** – Render complete UAV paths and obstacle environments
* **Flight Path Export** – `.txt` output format suitable for simulators or UAV systems

---

### Repository Structure

| Path                     | Description                                                                     |
| ------------------------ | ------------------------------------------------------------------------------- |
| `UAV_PathPlanning_code/` | Main Python scripts for 2D/3D path planning and visualization                   |
| `Slice_flatten/`         | Contains sliced 2D `.pcd` files (inside/outside obstacle & shooting point data) |
| `ImproveTime/`           | Time benchmarking comparison (`6HR` vs. `10Sec` visualizers)                    |
| `README.md`              | Project documentation                                                           |
| `*.txt`                  | Exported flight paths (x y z format)                                            |

---

### Algorithm Pipeline

#### 1. Preprocessing (C++)

* **`slice.cpp`**: Slices raw `.pcd` point cloud into evenly spaced horizontal layers
* Normalizes Z-coordinates to midpoint of each slice for layer consistency

#### 2. 2D Path Planning (Python)

* Generates occupancy grid at 0.1m resolution
* Applies morphological dilation + closing to seal wall gaps
* Shooting points are filtered if overlapping with obstacles
* KNN used for neighborhood graph, Kruskal's MST ensures shortest valid connections

#### 3. Multi-Layer Integration

* DFS with backtracking applied to each layer
* A\* is used to transition UAV between layers horizontally
* Vertical Z-paths connect endpoints across layers

#### 4. Visualization & Output

* `DEMO_3D_Visualize_ConnectLayer.py`: integrates all layers and draws full path
* Exported flight path in `.txt` for downstream applications or simulation

---

### Performance Boost

| Scenario     | Method                     | Runtime     |
| ------------ | -------------------------- | ----------- |
| Traditional  | Brute-force MST (n² check) | 6.4 hours   |
| This Project | KNN + obstacle filtering   | 6.9 seconds |

See `/ImproveTime/10Sec_3D_Visualize.py` vs. `6HR_3D_Visualize.py`

---

