# -*- coding: utf-8 -*-
"""
slice_and_visualize.py
======================
流程：
1. 讀取 config.json 參數
2. 將 obstacle / shooting 原始點雲切層
3. 自動以 *所有 shooting slice* 的最小 Z 為 z_base
4. 逐層 KNN + MST + 視覺化 (單層視窗 **保留**)
5. 跨層 DFS / A* / 垂直連線，整合 UAV 路徑
6. Open3D 顯示 3D 結果，並匯出 uav_path.txt

"""

from __future__ import annotations
import json, time, glob, os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import open3d as o3d
import cv2
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import heapq

# ---------------------------------------------------------------------------
# 讀取設定檔
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"設定檔 {path} 不存在")
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    # 基本檢查
    required_keys = [
        "slice_thickness",
        "offset",
        "obstacle_slice_dir",
        "shooting_slice_dir",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise KeyError(f"config.json 缺少欄位: {missing}")
    return cfg

# ---------------------------------------------------------------------------
# 切層 (由 C++ slice.cpp 改寫)
# ---------------------------------------------------------------------------

def slice_point_cloud(input_pcd: str | Path, output_dir: str | Path, thickness: float) -> None:
    """將單一點雲依 Z 設定切層並輸出 slice_i.pcd (Z 設為 slice 中點)"""
    input_pcd = Path(input_pcd)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cloud = o3d.io.read_point_cloud(str(input_pcd))
    if cloud.is_empty():
        print(f"[slice] ⚠️  {input_pcd} 為空，跳過切層")
        return
    points = np.asarray(cloud.points)
    min_z, max_z = points[:, 2].min(), points[:, 2].max()

    slice_idx = 0
    slice_start = min_z
    slice_end = slice_start + thickness

    while slice_start < max_z:
        mask = np.logical_and(points[:, 2] >= slice_start, points[:, 2] < slice_end)
        if mask.any():
            sliced = points[mask].copy()
            mid_z = (slice_start + slice_end) / 2.0
            sliced[:, 2] = mid_z  # 改寫 Z 為中點

            out_path = output_dir / f"slice_{slice_idx}.pcd"
            o3d.io.write_point_cloud(str(out_path), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sliced)))
            print(f"[slice] Saved {out_path}")
        slice_start = slice_end
        slice_end += thickness
        slice_idx += 1
    print("[slice] Completed!")

# ---------------------------------------------------------------------------
# 路徑規劃相關工具 (原樣貼入)
# ---------------------------------------------------------------------------

def apply_approx_knn(filtered_points: np.ndarray, x_scaled_shoot: np.ndarray, y_scaled_shoot: np.ndarray,
                     occupancy_grid_fixed: np.ndarray, resolution: float, max_neighbors: int = 15) -> nx.Graph:
    num_points = len(filtered_points)
    dist_matrix = np.full((num_points, num_points), np.inf)
    nbrs = NearestNeighbors(n_neighbors=min(max_neighbors + 1, num_points), algorithm="auto").fit(filtered_points)
    distances, indices = nbrs.kneighbors(filtered_points)
    grid_size_y, grid_size_x = occupancy_grid_fixed.shape

    for i in range(num_points):
        for j in indices[i][1:]:
            dist = distances[i][np.where(indices[i] == j)[0][0]]
            x1, y1 = x_scaled_shoot[i], y_scaled_shoot[i]
            x2, y2 = x_scaled_shoot[j], y_scaled_shoot[j]
            rr = np.linspace(y1, y2, num=max(2, int(dist / resolution))).astype(int)
            cc = np.linspace(x1, x2, num=max(2, int(dist / resolution))).astype(int)
            rr = np.clip(rr, 0, grid_size_y - 1)
            cc = np.clip(cc, 0, grid_size_x - 1)
            if np.any(occupancy_grid_fixed[rr, cc] == 1):
                continue
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    G = nx.Graph()
    for i in range(num_points):
        for j in range(num_points):
            if dist_matrix[i, j] < np.inf:
                G.add_edge(i, j, weight=dist_matrix[i, j])
    return G

# DFS & A* 保留原版實作

def dfs_with_backtracking(graph: nx.Graph, points_3d: np.ndarray, start_idx: int) -> List[List[float]]:
    visited = set()
    path = []
    def dfs(u):
        visited.add(u)
        path.append(points_3d[u])
        for v in sorted(list(graph.neighbors(u))):
            if v not in visited:
                dfs(v)
                path.append(points_3d[u])
    dfs(start_idx)
    return path

def dfs_traversal(graph: nx.Graph, start: int) -> List[int]:
    visited, order = set(), []
    def dfs(u):
        visited.add(u)
        order.append(u)
        for v in sorted(list(graph.neighbors(u))):
            if v not in visited:
                dfs(v)
    dfs(start)
    return order

def astar(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]
    rows, cols = grid.shape
    open_set: List[Tuple[float, Tuple[int, int]]] = []
    heapq.heappush(open_set, (0, start))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score = {start: 0.0}

    heuristic = lambda a, b: np.linalg.norm(np.array(a) - np.array(b))

    while open_set:
        current = heapq.heappop(open_set)[1]
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for d in neighbors:
            neighbor = (current[0] + d[0], current[1] + d[1])
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor[0], neighbor[1]] == 0:
                tentative_g = g_score[current] + heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
    return []

# ---------------------------------------------------------------------------
# 逐層處理 + 單層視覺化 (保留原行為)
# ---------------------------------------------------------------------------

def plan_layer_paths(obstacle_dir: Path, shooting_dir: Path, thickness: float, z_base: float, offset: int,
                     resolution: float = 0.1) -> Tuple[Dict[int, Dict[str, Any]], float, float, List[np.ndarray]]:
    layer_info: Dict[int, Dict[str, Any]] = {}
    total_mst_length = 0.0
    total_exec_time = 0.0
    all_obstacle_points: List[np.ndarray] = []

    slice_files = sorted(shooting_dir.glob("slice_*.pcd"), key=lambda p: int(p.stem.split("_")[-1]))
    layer_range = range(len(slice_files))

    for s_layer in tqdm(layer_range, desc="處理進度"):
        start_time = time.time()
        o_layer = s_layer + offset
        obs_path = obstacle_dir / f"slice_{o_layer}.pcd"
        shoot_path = shooting_dir / f"slice_{s_layer}.pcd"
        if not obs_path.exists() or not shoot_path.exists():
            print(f"❌ {obs_path} 或 {shoot_path} 不存在，跳過")
            continue

        obs_pcd = o3d.io.read_point_cloud(str(obs_path))
        shoot_pcd = o3d.io.read_point_cloud(str(shoot_path))
        obs_points = np.asarray(obs_pcd.points)
        shoot_points = np.asarray(shoot_pcd.points)
        if len(obs_points) == 0 or len(shoot_points) == 0:
            print("⚠️ 此層空點，跳過")
            continue

        z_value_obs = z_base + s_layer * thickness
        obs_points_3d = np.hstack((obs_points[:, :2], np.full((obs_points.shape[0], 1), z_value_obs)))
        all_obstacle_points.append(obs_points_3d)

        # Occupancy grid
        x_coords, y_coords = obs_points[:, 0], obs_points[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        grid_size_x = int((x_max - x_min) / resolution) + 1
        grid_size_y = int((y_max - y_min) / resolution) + 1
        occupancy_grid = np.zeros((grid_size_y, grid_size_x), dtype=np.uint8)
        x_scaled = ((x_coords - x_min) / resolution).astype(int)
        y_scaled = ((y_coords - y_min) / resolution).astype(int)
        x_scaled = np.clip(x_scaled, 0, grid_size_x - 1)
        y_scaled = np.clip(y_scaled, 0, grid_size_y - 1)
        occupancy_grid[y_scaled, x_scaled] = 1
        kernel = np.ones((5, 5), np.uint8)
        occupancy_grid_fixed = cv2.dilate(occupancy_grid, kernel, iterations=1)
        occupancy_grid_fixed = cv2.morphologyEx(occupancy_grid_fixed, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Filter shooting points
        x_shoot, y_shoot = shoot_points[:, 0], shoot_points[:, 1]
        x_scaled_shoot = ((x_shoot - x_min) / resolution).astype(int)
        y_scaled_shoot = ((y_shoot - y_min) / resolution).astype(int)
        x_scaled_shoot = np.clip(x_scaled_shoot, 0, grid_size_x - 1)
        y_scaled_shoot = np.clip(y_scaled_shoot, 0, grid_size_y - 1)
        mask = occupancy_grid_fixed[y_scaled_shoot, x_scaled_shoot] == 0
        filtered_points = shoot_points[mask]
        x_scaled_shoot = x_scaled_shoot[mask]
        y_scaled_shoot = y_scaled_shoot[mask]
        if len(filtered_points) < 2:
            print("⚠️ Shooting points 太少，跳過")
            continue

        # 建圖 + MST
        G = apply_approx_knn(filtered_points, x_scaled_shoot, y_scaled_shoot, occupancy_grid_fixed, resolution)
        MST = nx.minimum_spanning_tree(G)
        total_length = sum(nx.get_edge_attributes(MST, 'weight').values())
        exec_time = time.time() - start_time
        print(f"📏 Layer {s_layer} MST 長度 {total_length:.2f} m, ⏱ {exec_time:.2f} s")

        total_mst_length += total_length
        total_exec_time += exec_time

        z_value = z_base + s_layer * thickness
        filtered_points_3d = np.hstack((filtered_points[:, :2], np.full((len(filtered_points), 1), z_value)))

        # --- 單層視覺化（保留原行為） ---
        layer_points = o3d.geometry.PointCloud()
        layer_points.points = o3d.utility.Vector3dVector(filtered_points_3d)
        layer_points.paint_uniform_color([1, 0, 0])
        layer_lines = o3d.geometry.LineSet()
        layer_lines.points = o3d.utility.Vector3dVector(filtered_points_3d)
        layer_lines.lines = o3d.utility.Vector2iVector([[i, j] for i, j in MST.edges()])
        layer_lines.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(MST.edges()))
        obstacle_cloud = o3d.geometry.PointCloud()
        obstacle_cloud.points = o3d.utility.Vector3dVector(obs_points_3d)
        obstacle_cloud.paint_uniform_color([0, 0, 0])

        o3d.visualization.draw_geometries(
            [layer_points, layer_lines, obstacle_cloud],
            window_name=f"第 {s_layer} 層 - 規劃結果",
            width=1000,
            height=800,
        )

        layer_info[s_layer] = {
            "filtered_points": filtered_points[:, :2],
            "points_3d": filtered_points_3d,
            "mst_graph": MST,
            "occupancy_grid": occupancy_grid_fixed,
            "x_min": x_min,
            "y_min": y_min,
            "resolution": resolution,
            "grid_size": (grid_size_x, grid_size_y)
        }

    return layer_info, total_mst_length, total_exec_time, all_obstacle_points

# ---------------------------------------------------------------------------
# 跨層整合路徑
# ---------------------------------------------------------------------------

def integrate_uav_path(layer_info: Dict[int, Dict[str, Any]], thickness: float, z_base: float) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    dfs_segments, astar_segments, vertical_segments = [], [], []
    if not layer_info:
        return np.empty((0, 3)), dfs_segments, astar_segments, vertical_segments

    sorted_layers = sorted(layer_info.keys())
    top_layer = sorted_layers[0]
    data = layer_info[top_layer]
    points_2d = data["filtered_points"]
    start_idx = np.lexsort((points_2d[:, 1], points_2d[:, 0]))[0]
    layer_path = dfs_with_backtracking(data["mst_graph"], data["points_3d"], start_idx)
    dfs_segments.append(layer_path)
    prev_end = layer_path[-1]

    for k_idx, layer in enumerate(sorted_layers[1:]):
        data_next = layer_info[layer]
        points_2d_next = data_next["filtered_points"]
        vertical_proj = prev_end[:2]
        distances = np.linalg.norm(points_2d_next - vertical_proj, axis=1)
        candidate_idx = int(np.argmin(distances))

        # A* 平面路徑
        x_min, y_min = data_next["x_min"], data_next["y_min"]
        res = data_next["resolution"]
        grid_size_x, grid_size_y = data_next["grid_size"]
        start_grid = (int((prev_end[1] - y_min) / res), int((prev_end[0] - x_min) / res))
        cand_pt = points_2d_next[candidate_idx]
        goal_grid = (int((cand_pt[1] - y_min) / res), int((cand_pt[0] - x_min) / res))
        grid_next = data_next["occupancy_grid"]
        path_grid = astar(grid_next, start_grid, goal_grid)
        if not path_grid:
            transition_path_flat = [[prev_end[0], prev_end[1], prev_end[2]], [cand_pt[0], cand_pt[1], prev_end[2]]]
        else:
            path_world = [[c * res + x_min, r * res + y_min] for r, c in path_grid]
            transition_path_flat = [[pt[0], pt[1], prev_end[2]] for pt in path_world]
        astar_segments.append(transition_path_flat)

        cand_pt_3d = data_next["points_3d"][candidate_idx]
        vertical_segments.append([[prev_end[0], prev_end[1], prev_end[2]], cand_pt_3d.tolist()])

        order_next = dfs_traversal(data_next["mst_graph"], candidate_idx)
        layer_path_next = data_next["points_3d"][order_next, :]
        dfs_segments.append(layer_path_next.tolist())
        prev_end = layer_path_next[-1]

    # --- 整合所有段為 numpy full_path ---
    full_path = []
    for seg_group in (dfs_segments, astar_segments, vertical_segments):
        for seg in seg_group:
            full_path.extend(seg)
    return np.array(full_path), dfs_segments, astar_segments, vertical_segments

# ---------------------------------------------------------------------------
# 視覺化 3D & 匯出 txt
# ---------------------------------------------------------------------------

def visualize_and_export(layer_info: Dict[int, Dict[str, Any]], full_path: np.ndarray, obstacle_all: List[np.ndarray],
                          work_dir: Path, total_len: float, total_time: float) -> None:
    if full_path.size == 0:
        print("⚠️ 沒有有效路徑可顯示")
        return

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(full_path)
    line_set.lines = o3d.utility.Vector2iVector(np.array([[i, i + 1] for i in range(len(full_path) - 1)]))
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * (len(full_path) - 1))

    pcd_points = o3d.geometry.PointCloud()
    pcd_points.points = o3d.utility.Vector3dVector(full_path)
    pcd_points.paint_uniform_color([1, 0, 0])

    obstacle_all_pts = np.vstack(obstacle_all)
    obstacle_cloud = o3d.geometry.PointCloud()
    obstacle_cloud.points = o3d.utility.Vector3dVector(obstacle_all_pts)
    obstacle_cloud.paint_uniform_color([0, 0, 0])

    o3d.visualization.draw_geometries(
        [obstacle_cloud, pcd_points, line_set],
        window_name="整合 3D 與 UAV 飛行路徑視覺化",
        width=1200,
        height=900,
    )

    # 匯出 txt
    work_dir.mkdir(parents=True, exist_ok=True)
    output_txt = work_dir / "uav_path.txt"
    np.savetxt(output_txt, full_path, fmt="%.6f", delimiter=" ")
    print(f"路徑已匯出 {output_txt}")
    print(f"總 MST 長度: {total_len:.2f} m, ⏱️ 總時間: {total_time:.2f} s")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config("config.json")
    thickness = float(cfg["slice_thickness"])
    offset = int(cfg["offset"])

    obstacle_slice_dir = Path(cfg["obstacle_slice_dir"])
    shooting_slice_dir = Path(cfg["shooting_slice_dir"])

    # --- optional slicing ---
    if cfg.get("obstacle_raw_pcd"):
        slice_point_cloud(cfg["obstacle_raw_pcd"], obstacle_slice_dir, thickness)
    if cfg.get("shooting_raw_pcd"):
        slice_point_cloud(cfg["shooting_raw_pcd"], shooting_slice_dir, thickness)

    # --- calc z_base from ALL shooting slice files ---
    shoot_files = list(shooting_slice_dir.glob("slice_*.pcd"))
    if not shoot_files:
        raise FileNotFoundError("No shooting slice_*.pcd found")
    min_z_vals = []
    for f in shoot_files:
        pts = np.asarray(o3d.io.read_point_cloud(str(f)).points)
        if pts.size:
            min_z_vals.append(pts[:, 2].min())
    z_base = float(np.min(min_z_vals))
    print(f"[基準 Z] shooting 全域最小 Z = {z_base:.3f}")

    layer_info, total_len, total_time, obstacle_all_pts = plan_layer_paths(
        obstacle_slice_dir, shooting_slice_dir, thickness, z_base, offset)

    full_path, *_ = integrate_uav_path(layer_info, thickness, z_base)

    visualize_and_export(layer_info, full_path, obstacle_all_pts, Path("./"), total_len, total_time)


if __name__ == "__main__":
    main()
