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
import networkx as nx
from sklearn.neighbors import NearestNeighbors  # 放最上面就好，一次即可
from pathlib import Path

def dfs_with_backtracking(graph, points_3d, start_idx):
    visited = set()
    path = []

    def dfs(u):
        visited.add(u)
        path.append(points_3d[u])
        for v in sorted(list(graph.neighbors(u))):
            if v not in visited:
                dfs(v)
                path.append(points_3d[u])  # 退回原本的點

    dfs(start_idx)
    return path


def apply_approx_knn(filtered_points, x_scaled_shoot, y_scaled_shoot, occupancy_grid_fixed, resolution, max_neighbors=15):
    """
    使用近似 KNN 建圖來取代全配對暴力法，過濾穿越障礙物的邊。
    
    :param filtered_points: 經過障礙物濾除的 shooting points (n x 3)
    :param x_scaled_shoot, y_scaled_shoot: 對應的格點索引位置
    :param occupancy_grid_fixed: 修正後的障礙物網格
    :param resolution: 每格解析度
    :param max_neighbors: 每個點最多找幾個近鄰
    :return: NetworkX Graph 僅包含障礙物可通過的近鄰邊
    """
    num_points = len(filtered_points)
    dist_matrix = np.full((num_points, num_points), np.inf)
    
    # 建立近鄰搜尋器
    nbrs = NearestNeighbors(n_neighbors=min(max_neighbors + 1, num_points), algorithm='auto').fit(filtered_points)
    distances, indices = nbrs.kneighbors(filtered_points)
    
    grid_size_y, grid_size_x = occupancy_grid_fixed.shape
    
    for i in range(num_points):
        for j in indices[i][1:]:  # indices[i][0] 是自己
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

    # 建圖
    G = nx.Graph()
    for i in range(num_points):
        for j in range(num_points):
            if dist_matrix[i, j] < np.inf:
                G.add_edge(i, j, weight=dist_matrix[i, j])
    return G


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
# shooting_dir = "/home/wasn/Desktop/Project/charlie/Slice_flatten/output"
# shooting_dir = "/home/wasn/Desktop/Project/charlie/Slice_flatten/dis_25_inside"
# shooting_dir = "/home/wasn/Desktop/Project/charlie/Slice_flatten/newnew_inside"
shooting_dir = "/home/wasn/Desktop/Project/charlie/Slice_flatten/KD_2000_dis1_1"
# shooting_dir = "/home/wasn/Desktop/Project/charlie/Slice_flatten/KD_2000_dis15_15"
# shooting_dir = "/home/wasn/Desktop/Project/charlie/Slice_flatten/KD_2000_dis2_1"

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

    if len(filtered_points) > 15000:
        print(f"⏩ 第 {layer} 層 Shooting Points 數量為{len(filtered_points)}，因為點數過多，跳過")
        continue 
    if len(filtered_points) < 2:
        print("⚠️ 有效 Shooting Points 太少，跳過")
        continue

    print(f"✅ 有效 Shooting Points 數: {len(filtered_points)}")

    # num_points = len(filtered_points)
    # dist_matrix = np.full((num_points, num_points), np.inf)
    # for i in range(num_points):
    #     for j in range(i + 1, num_points):
    #         dist = euclidean(filtered_points[i], filtered_points[j])
    #         x1, y1 = x_scaled_shoot[i], y_scaled_shoot[i]
    #         x2, y2 = x_scaled_shoot[j], y_scaled_shoot[j]
    #         rr = np.linspace(y1, y2, num=max(2, int(dist / resolution))).astype(int)
    #         cc = np.linspace(x1, x2, num=max(2, int(dist / resolution))).astype(int)
    #         rr = np.clip(rr, 0, grid_size_y - 1)
    #         cc = np.clip(cc, 0, grid_size_x - 1)
    #         if np.any(occupancy_grid_fixed[rr, cc] == 1):
    #             continue
    #         dist_matrix[i, j] = dist_matrix[j, i] = dist

    # G = nx.Graph()
    # for i in range(num_points):
    #     for j in range(i + 1, num_points):
    #         if dist_matrix[i, j] < np.inf:
    #             G.add_edge(i, j, weight=dist_matrix[i, j])

    if len(filtered_points) < 2:
        continue

    
    # 執行近似 KNN 建圖（取代暴力）
    num_points = len(filtered_points)

    G = apply_approx_knn(
        filtered_points,
        x_scaled_shoot,
        y_scaled_shoot,
        occupancy_grid_fixed,
        resolution,
        max_neighbors=15  # 可調整近鄰數量
    )

    MST = nx.minimum_spanning_tree(G)
    total_length = sum(nx.get_edge_attributes(MST, 'weight').values())
    print(f"📏 MST 總長度: {total_length:.2f} 公尺")

    end_time = time.time()  # ✅ 新增
    elapsed_time = end_time - start_time
    print(f"⏱️ 執行時間：{elapsed_time:.2f} 秒")  # ✅ 新增

    z_value =150.0 + layer * 1.0
    filtered_points_3d = np.hstack((filtered_points[:, :2], np.full((num_points, 1), z_value)))

    # --- 顯示單層結果用 ---
    layer_points = o3d.geometry.PointCloud()
    layer_points.points = o3d.utility.Vector3dVector(filtered_points_3d)
    layer_points.paint_uniform_color([1, 0, 0])  # 紅色點

    layer_lines = o3d.geometry.LineSet()
    layer_lines.points = o3d.utility.Vector3dVector(filtered_points_3d)
    layer_lines.lines = o3d.utility.Vector2iVector([[i, j] for i, j in MST.edges()])
    layer_lines.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(MST.edges()))  # 綠色邊

    obstacle_cloud = o3d.geometry.PointCloud()
    obstacle_cloud.points = o3d.utility.Vector3dVector(obs_points_3d)
    obstacle_cloud.paint_uniform_color([0, 0, 0])  # 黑色障礙物

    o3d.visualization.draw_geometries(
        [layer_points, layer_lines, obstacle_cloud],
        window_name=f"第 {layer} 層 - 規劃結果",
        width=1000,
        height=800,
    )

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
        layer_path = dfs_with_backtracking(data["mst_graph"], data["points_3d"], start_idx)
        dfs_segments.append(layer_path)
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

            cand_pt_3d = data_next["points_3d"][candidate_idx]
            vertical_transition = [
                [prev_end[0], prev_end[1], prev_end[2]],
                cand_pt_3d.tolist()  # ✅ 直接抓有正確 Z 的點
            ]


            transition_segments_astar.append(transition_path_flat)
            transition_segments_vertical.append(vertical_transition)

            if candidate_idx not in data_next["mst_graph"].nodes:
                # 找一個最近且有在 MST nodes 裡的點
                distances_sorted_idx = np.argsort(distances)
                for idx in distances_sorted_idx:
                    if idx in data_next["mst_graph"].nodes:
                        candidate_idx = idx
                        break
                else:
                    print(f"❌ 找不到可用的起點於 MST (Layer {layer})，跳過這層")
                    continue  # 直接跳過這層

            order_next = dfs_traversal(data_next["mst_graph"], candidate_idx)
            layer_path_next = data_next["points_3d"][order_next, :]
            dfs_segments.append(layer_path_next.tolist())
            prev_end = layer_path_next[-1]
    
    def create_lineset_from_segments(segments, color):
        all_points = []
        all_edges = []
        point_offset = 0
        for seg in segments:
            # 過濾掉 z < 150 的點
            seg_filtered = [pt for pt in seg if pt[2] >= 150.0]
            if len(seg_filtered) < 2:
                continue
            seg_edges = [[i + point_offset, i + 1 + point_offset] for i in range(len(seg_filtered) - 1)]
            all_points.extend(seg_filtered)
            all_edges.extend(seg_edges)
            point_offset += len(seg_filtered)

        # 🛡️ 若沒有任何點，回傳空 LineSet 避免崩潰
        if len(all_points) == 0 or len(all_edges) == 0:
            return o3d.geometry.LineSet()

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.array(all_points))
        ls.lines = o3d.utility.Vector2iVector(np.array(all_edges))
        ls.colors = o3d.utility.Vector3dVector([color] * len(all_edges))
        return ls



    dfs_line_set = create_lineset_from_segments(dfs_segments, [0, 1, 0])
    astar_line_set = create_lineset_from_segments(transition_segments_astar, [0, 1, 0])
    vertical_line_set = create_lineset_from_segments(transition_segments_vertical, [0, 1, 0])

    print("\n📊 所有層總 MST 長度：{:.2f} 公尺".format(total_mst_length))
    print("⏱️ 所有層總執行時間：{:.2f} 秒".format(total_exec_time))

        # === 匯出最終完整飛行路徑 ===

    # 收集所有飛行段，並過濾 z < 150 的點
    full_path = []

    for seg in dfs_segments:
        full_path.extend([pt for pt in seg if pt[2] >= 150.0])

    for seg in transition_segments_astar:
        full_path.extend([pt for pt in seg if pt[2] >= 150.0])

    for seg in transition_segments_vertical:
        full_path.extend([pt for pt in seg if pt[2] >= 150.0])


    # 轉成np array
    full_path_np = np.array(full_path)

    # 儲存成txt檔
    # save_path = Path("/home/wasn/Desktop/Project/charlie/1_DEMO_code/uav_flight_path_KD_2000_dis1_1.txt")
    # save_path = Path("/home/wasn/Desktop/Project/charlie/1_DEMO_code/uav_flight_path_KD_2000_dis15_15.txt")
    save_path = Path("/home/wasn/Desktop/Project/charlie/1_DEMO_code/uav_flight_path_KD_2000_dis2_1.txt")
    np.savetxt(save_path, full_path_np, fmt="%.6f", delimiter=" ")

    print(f"✅ 飛行路徑已儲存到：{save_path}")


    geometries = [obstacle_all, pcd_points, dfs_line_set, astar_line_set, vertical_line_set]

# 先計算所有物件的中心點 (Centroid)
all_points = []
for geo in geometries:
    if hasattr(geo, 'points'):
        all_points.append(np.asarray(geo.points))
if all_points:
    all_points = np.vstack(all_points)
    center = np.mean(all_points, axis=0)
else:
    center = np.array([0, 0, 0])

# 建立 Visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="整合 3D 與 UAV 飛行路徑視覺化", width=1200, height=900)

for geo in geometries:
    vis.add_geometry(geo)

# 設定居中視角
view_ctl = vis.get_view_control()
view_ctl.set_lookat(center)
view_ctl.set_zoom(0.5)  # 可以微調，越小越近
view_ctl.set_front([0, 0, -1])  # (選擇性) 正上往下看
view_ctl.set_up([0, 1, 0])      # (選擇性) Y軸朝上

# 顯示
vis.run()
vis.destroy_window()
