import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import networkx as nx
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import time
import os
import heapq  # --- 新增：A* 演算法需要用到

# --- 新增：DFS Traversal (深度優先搜尋) 函數 ---
def dfs_traversal(graph, start):
    visited = set()
    order = []
    def dfs(u):
        visited.add(u)
        order.append(u)
        for v in sorted(list(graph.neighbors(u))):
            if v not in visited:
                dfs(v)
    dfs(start)
    return order

# --- 新增：A* 演算法函數 ---
def astar(grid, start, goal):
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

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
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor[0], neighbor[1]] == 1:
                    continue
                tentative_g_score = g_score[current] + heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
    return []

# === 參數設定 ===
resolution = 0.1
layer_range = range(16)
obstacle_dir = "/home/wasn/Desktop/Project/charlie/Slice_flatten/new_outside"
shooting_dir = "/home/wasn/Desktop/Project/charlie/Slice_flatten/output"

all_3d_points = []
all_edges = []
global_index = 0
all_obstacle_points = []
layer_info = {}
total_mst_length = 0.0  # 總 MST 長度
total_exec_time = 0.0   # 總執行時間


for layer in tqdm(layer_range, desc="處理進度"):
    print(f"\n📂 [處理第 {layer} 層]")
    start_time = time.time()  # ✅ 新增

    obs_path = os.path.join(obstacle_dir, f"slice_{layer}.pcd")
    shoot_path = os.path.join(shooting_dir, f"slice_{layer}.pcd")

    if not os.path.exists(obs_path) or not os.path.exists(shoot_path):
        print("❌ 檔案不存在，跳過")
        continue

    obs_pcd = o3d.io.read_point_cloud(obs_path)
    shoot_pcd = o3d.io.read_point_cloud(shoot_path)
    obs_points = np.asarray(obs_pcd.points)
    shoot_points = np.asarray(shoot_pcd.points)

    if len(obs_points) == 0 or len(shoot_points) == 0:
        print("⚠️ 此層點數為空，跳過")
        continue

    z_value_obs =150.0 + layer * 1.0
    obs_points_3d = np.hstack((obs_points[:, :2], np.full((obs_points.shape[0], 1), z_value_obs)))
    all_obstacle_points.append(obs_points_3d)

    x_coords = obs_points[:, 0]
    y_coords = obs_points[:, 1]
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

    x_shoot = shoot_points[:, 0]
    y_shoot = shoot_points[:, 1]
    x_scaled_shoot = ((x_shoot - x_min) / resolution).astype(int)
    y_scaled_shoot = ((y_shoot - y_min) / resolution).astype(int)
    x_scaled_shoot = np.clip(x_scaled_shoot, 0, grid_size_x - 1)
    y_scaled_shoot = np.clip(y_scaled_shoot, 0, grid_size_y - 1)
    mask = occupancy_grid_fixed[y_scaled_shoot, x_scaled_shoot] == 0
    filtered_points = shoot_points[mask]
    x_scaled_shoot = x_scaled_shoot[mask]
    y_scaled_shoot = y_scaled_shoot[mask]

    if len(filtered_points) > 2000:
        print(f"⏩ 第 {layer} 層 Shooting Points 數量為{len(filtered_points)}，因為點數過多，跳過")
        continue 
    if len(filtered_points) < 2:
        print("⚠️ 有效 Shooting Points 太少，跳過")
        continue

    print(f"✅ 有效 Shooting Points 數: {len(filtered_points)}")

    num_points = len(filtered_points)
    dist_matrix = np.full((num_points, num_points), np.inf)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = euclidean(filtered_points[i], filtered_points[j])
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
        for j in range(i + 1, num_points):
            if dist_matrix[i, j] < np.inf:
                G.add_edge(i, j, weight=dist_matrix[i, j])
    MST = nx.minimum_spanning_tree(G)
    total_length = sum(nx.get_edge_attributes(MST, 'weight').values())
    print(f"📏 MST 總長度: {total_length:.2f} 公尺")

    end_time = time.time()  # ✅ 新增
    elapsed_time = end_time - start_time
    print(f"⏱️ 執行時間：{elapsed_time:.2f} 秒")  # ✅ 新增

    z_value =150.0 + layer * 1.0
    filtered_points_3d = np.hstack((filtered_points[:, :2], np.full((num_points, 1), z_value)))
    all_3d_points.extend(filtered_points_3d.tolist())
    for i, j in MST.edges():
        all_edges.append([global_index + i, global_index + j])
    global_index += num_points

    total_mst_length += total_length     # 累加該層長度
    total_exec_time += elapsed_time      # 累加該層時間

    
    layer_info[layer] = {
        "filtered_points": filtered_points[:, :2],
        "points_3d": filtered_points_3d,
        "mst_graph": MST,
        "occupancy_grid": occupancy_grid_fixed,
        "x_min": x_min,
        "y_min": y_min,
        "resolution": resolution,
        "grid_size": (grid_size_x, grid_size_y)
    }

if len(all_3d_points) > 0 and len(all_edges) > 0:
    print("\n🧱 整合所有層為 3D MST 視覺化...")
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(all_3d_points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(all_edges))
    line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1]] * len(all_edges))

    pcd_points = o3d.geometry.PointCloud()
    pcd_points.points = o3d.utility.Vector3dVector(np.array(all_3d_points))
    pcd_points.paint_uniform_color([1, 0, 0])

    obstacle_all = o3d.geometry.PointCloud()
    obstacle_all.points = o3d.utility.Vector3dVector(np.vstack(all_obstacle_points))
    obstacle_all.paint_uniform_color([0, 0, 0])

    # --- UAV 路徑整合（DFS + A* + Z 軸 transition） ---
    dfs_segments = []
    transition_segments_astar = []
    transition_segments_vertical = []

    if layer_info:
        sorted_layers = sorted(layer_info.keys(), reverse=True)
        top_layer = sorted_layers[0]
        data = layer_info[top_layer]
        points_2d = data["filtered_points"]
        start_idx = np.lexsort((points_2d[:,1], points_2d[:,0]))[0]
        order = dfs_traversal(data["mst_graph"], start_idx)
        layer_path = data["points_3d"][order, :]
        dfs_segments.append(layer_path.tolist())
        prev_end = layer_path[-1]

        for layer in sorted_layers[1:]:
            data_next = layer_info[layer]
            points_2d_next = data_next["filtered_points"]
            vertical_proj = prev_end[:2]
            distances = np.linalg.norm(points_2d_next - vertical_proj, axis=1)
            candidate_idx = np.argmin(distances)

            x_min_next = data_next["x_min"]
            y_min_next = data_next["y_min"]
            res = data_next["resolution"]
            grid_size_x_next, grid_size_y_next = data_next["grid_size"]
            start_grid = ( int((prev_end[1] - y_min_next)/res), int((prev_end[0] - x_min_next)/res) )
            cand_pt = points_2d_next[candidate_idx]
            goal_grid = ( int((cand_pt[1] - y_min_next)/res), int((cand_pt[0] - x_min_next)/res) )
            grid_next = data_next["occupancy_grid"]
            path_grid = astar(grid_next, start_grid, goal_grid)
            if not path_grid:
                print(f"⚠️ Layer {layer} A* 未找到從 {start_grid} 到 {goal_grid} 的路徑，直接連線")
                transition_path_flat = [ [prev_end[0], prev_end[1], layer*1.0] , [cand_pt[0], cand_pt[1], layer*1.0] ]
            else:
                path_world = []
                for (r, c) in path_grid:
                    x_world = c * res + x_min_next
                    y_world = r * res + y_min_next
                    path_world.append([x_world, y_world])
                transition_path_flat = [[pt[0], pt[1], layer*1.0] for pt in path_world]

            vertical_transition = [
                [prev_end[0], prev_end[1], prev_end[2]],
                [cand_pt[0], cand_pt[1], layer*1.0]
            ]

            transition_segments_astar.append(transition_path_flat)
            transition_segments_vertical.append(vertical_transition)

            order_next = dfs_traversal(data_next["mst_graph"], candidate_idx)
            layer_path_next = data_next["points_3d"][order_next, :]
            dfs_segments.append(layer_path_next.tolist())
            prev_end = layer_path_next[-1]

    def create_lineset_from_segments(segments, color):
        all_points = []
        all_edges = []
        point_offset = 0
        for seg in segments:
            if len(seg) < 2:
                continue
            seg_edges = [[i + point_offset, i + 1 + point_offset] for i in range(len(seg) - 1)]
            all_points.extend(seg)
            all_edges.extend(seg_edges)
            point_offset += len(seg)
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.array(all_points))
        ls.lines = o3d.utility.Vector2iVector(np.array(all_edges))
        ls.colors = o3d.utility.Vector3dVector([color] * len(all_edges))
        return ls

    dfs_line_set = create_lineset_from_segments(dfs_segments, [0, 1, 0])
    astar_line_set = create_lineset_from_segments(transition_segments_astar, [1, 1, 0])
    vertical_line_set = create_lineset_from_segments(transition_segments_vertical, [0, 1, 1])

    print("\n📊 所有層總 MST 長度：{:.2f} 公尺".format(total_mst_length))
    print("⏱️ 所有層總執行時間：{:.2f} 秒".format(total_exec_time))

    o3d.visualization.draw_geometries(
        [obstacle_all, pcd_points, line_set, dfs_line_set, astar_line_set, vertical_line_set],
        window_name="整合 3D 與 UAV 飛行路徑視覺化",
        width=1200, height=900
    )
else:
    print("⚠️ 無有效層可整合為 3D 線條圖。")


