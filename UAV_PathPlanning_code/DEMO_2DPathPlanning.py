import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import networkx as nx
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors
import time

start_time = time.time()

# === 1. 讀取 PCD 檔案，並投影為 2D 平面 (障礙物) ===
#pcd_path = "/home/wasn/Desktop/Project/charlie/Slice_flatten/new_outside/slice_13.pcd"
pcd_path = "/home/wasn/Desktop/Project/charlie/Slice_flatten/new_outside/slice_10.pcd"
pcd = o3d.io.read_point_cloud(pcd_path)

# 取得點雲的 XYZ 座標
points = np.asarray(pcd.points)
x_coords = points[:, 0]
y_coords = points[:, 1]

# === 2. 設定解析度與計算 1:1 網格大小 ===
resolution = 0.1  # 每個網格代表 0.1m
x_min, x_max = np.min(x_coords), np.max(x_coords)
y_min, y_max = np.min(y_coords), np.max(y_coords)

grid_size_x = int((x_max - x_min) / resolution) + 1
grid_size_y = int((y_max - y_min) / resolution) + 1

# 建立 2D 佔據網格 (0=自由, 1=障礙物)
occupancy_grid = np.zeros((grid_size_y, grid_size_x), dtype=np.uint8)

# === 3. 轉換障礙物點到 2D 網格 ===
x_scaled = ((x_coords - x_min) / resolution).astype(int)
y_scaled = ((y_coords - y_min) / resolution).astype(int)

# 限制索引範圍
x_scaled = np.clip(x_scaled, 0, grid_size_x - 1)
y_scaled = np.clip(y_scaled, 0, grid_size_y - 1)

# 設定障礙物
occupancy_grid[y_scaled, x_scaled] = 1  # 1 表示障礙物

# === 4. 顯示 **原始** 障礙物地圖 ===
plt.figure(figsize=(8, 8))
plt.imshow(occupancy_grid, cmap="gray_r", origin="lower")  # 原始障礙物地圖
plt.title("Original Occupancy Grid (Before Processing)")
plt.xlabel("X Axis (meters)")
plt.ylabel("Y Axis (meters)")
plt.show()

# === 5. 修補邊界與漏洞 (膨脹 + 閉運算) ===
kernel = np.ones((5, 5), np.uint8)  # 設定 Kernel 大小
occupancy_grid_fixed = cv2.dilate(occupancy_grid, kernel, iterations=1)  # 先膨脹
occupancy_grid_fixed = cv2.morphologyEx(occupancy_grid_fixed, cv2.MORPH_CLOSE, kernel, iterations=1)  # 再閉運算

# === 6. 顯示 **修正後** 的障礙物地圖 ===
plt.figure(figsize=(8, 8))
plt.imshow(occupancy_grid_fixed, cmap="gray_r", origin="lower")  # 修正後的障礙物地圖
plt.title("Fixed Occupancy Grid (After Dilation + Closing)")
plt.xlabel("X Axis (meters)")
plt.ylabel("Y Axis (meters)")
plt.show()

# === 7. 讀取 Shooting Points (內部掃描點) ===
#shooting_pcd_path = "/home/wasn/Desktop/Project/charlie/Slice_flatten/dis_25_inside/slice_10.pcd"
#shooting_pcd_path = "/home/wasn/Desktop/Project/charlie/Slice_flatten/new_inside/slice_13.pcd"
#shooting_pcd_path = "/home/wasn/Desktop/Project/charlie/Slice_flatten/newnew_inside/slice_6.pcd"
shooting_pcd_path = "/home/wasn/Desktop/Project/charlie/Slice_flatten/KD_2000_dis15_15/slice_10.pcd"
# shooting_pcd_path = "/home/wasn/Desktop/Project/charlie/Slice_flatten/newnew_inside/slice_13.pcd"
shooting_pcd = o3d.io.read_point_cloud(shooting_pcd_path)
shooting_points = np.asarray(shooting_pcd.points)

# 轉換 Shooting Points 到 2D 網格
x_shooting = shooting_points[:, 0]
y_shooting = shooting_points[:, 1]

x_scaled_shooting = ((x_shooting - x_min) / resolution).astype(int)
y_scaled_shooting = ((y_shooting - y_min) / resolution).astype(int)

# 限制索引範圍
x_scaled_shooting = np.clip(x_scaled_shooting, 0, grid_size_x - 1)
y_scaled_shooting = np.clip(y_scaled_shooting, 0, grid_size_y - 1)


# === 8. 建立 KNN 邊並檢查障礙物 ===
num_points = len(shooting_points)
x_shooting = shooting_points[:, 0]
y_shooting = shooting_points[:, 1]
x_scaled_shooting = ((x_shooting - x_min) / resolution).astype(int)
y_scaled_shooting = ((y_shooting - y_min) / resolution).astype(int)
x_scaled_shooting = np.clip(x_scaled_shooting, 0, grid_size_x - 1)
y_scaled_shooting = np.clip(y_scaled_shooting, 0, grid_size_y - 1)

# 使用 KNN 找鄰居
k = 5  # 每個點找最近 3 個鄰居
nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(shooting_points[:, :2])  # 只用 XY 投影做鄰近
distances, indices = nbrs.kneighbors(shooting_points[:, :2])

# 新增：建立合法邊
edges = []
for i in range(num_points):
    for j in range(1, k+1):  # j=0 是自己，跳過
        neighbor_idx = indices[i, j]
        dist = distances[i, j]

        # 計算在網格上的直線路徑
        x1, y1 = x_scaled_shooting[i], y_scaled_shooting[i]
        x2, y2 = x_scaled_shooting[neighbor_idx], y_scaled_shooting[neighbor_idx]
        rr = np.linspace(y1, y2, num=max(2, int(dist / resolution))).astype(int)
        cc = np.linspace(x1, x2, num=max(2, int(dist / resolution))).astype(int)
        rr = np.clip(rr, 0, grid_size_y - 1)
        cc = np.clip(cc, 0, grid_size_x - 1)

        # 路徑上不能有障礙物
        if np.any(occupancy_grid_fixed[rr, cc] == 1):
            continue
        edges.append((i, neighbor_idx, dist))

# === 9. 使用 Kruskal 在合法邊中找出 MST ===
import networkx as nx

G = nx.Graph()
for i, j, dist in edges:
    G.add_edge(i, j, weight=dist)

MST = nx.minimum_spanning_tree(G)

# === 10. 視覺化最終 MST 結果 ===
plt.figure(figsize=(8, 8))
plt.imshow(occupancy_grid_fixed, cmap="gray_r", origin="lower")
plt.scatter(x_scaled_shooting, y_scaled_shooting, c='red', s=10, label="Shooting Points")

for edge in MST.edges():
    i, j = edge
    plt.plot([x_scaled_shooting[i], x_scaled_shooting[j]], 
             [y_scaled_shooting[i], y_scaled_shooting[j]], color='blue', linewidth=1.5)

end_time = time.time()
print(f"⏱️ 執行時間：{end_time - start_time:.3f} 秒")

plt.title("KNN Filtered + MST Path (Obstacle-Free)")
plt.xlabel("X Axis (meters)")
plt.ylabel("Y Axis (meters)")
plt.legend()
plt.show()
