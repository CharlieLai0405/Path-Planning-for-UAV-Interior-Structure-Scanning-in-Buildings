# UAV 多層點雲自動路徑規劃系統

本專案提供一鍵執行的 Python 腳本，可將原始室內外點雲自動分層、套用障礙物排除、建立各層最小生成樹、跨層整合路徑，最終產生可視化的 3D 飛行路線與 `uav_path.txt` 路徑檔案。

## 專案結構

```
.
├── config.json              ← 所有參數設定與路徑配置
├── slice_and_visualize.py   ← 主程式（點兩下或用指令執行）
├── new_outside/             ← 切層後的障礙物資料夾（自動建立）
├── merged_all_points/       ← 切層後的拍攝點資料夾（自動建立）
└── uav_path.txt             ← 輸出的飛行路徑（每行為 x y z）
```

---

## 第一步：編輯 `config.json`

請先確認以下參數路徑已對應到正確位置（可用絕對路徑或相對路徑）：

```json
{
  "slice_thickness": 2.0,
  "offset": 1,

  "obstacle_raw_pcd": "資料路徑/Indoor_and_outdoor.pcd",
  "shooting_raw_pcd": "資料路徑/merged_all_points.pcd",

  "obstacle_slice_dir": "CHT_DEMO_VisualEveryLayer/new_outside",
  "shooting_slice_dir": "CHT_DEMO_VisualEveryLayer/merged_all_points"
}
```

* `slice_thickness`：每層厚度（公尺）
* `offset`：障礙物切層偏移量（通常設為 1）
* `*_raw_pcd`：原始 PCD 點雲檔（必填）
* `*_slice_dir`：切層後的輸出資料夾

---

## 第二步：執行主程式

### 方式一：在終端機執行

```bash
python slice_and_visualize.py
```

### 方式二：點兩下 `slice_and_visualize.py` 執行

---

## 第三步：自動流程說明

執行時會自動完成以下步驟：

1. **切層**：根據 `slice_thickness`，將 obstacle 和 shooting PCD 分層
2. **計算基準高度**：以所有拍攝點切層的最小 Z 當作 `z_base`
3. **逐層處理**：

   * 建立障礙物佔據圖（occupancy grid）
   * 濾除障礙區域的 shooting points
   * 以 KNN + MST 建立路徑圖
   * 顯示每層 2D 規劃圖
4. **跨層整合**：

   * DFS 建構每層內部路徑
   * A\* 與垂直線連接跨層點
5. **3D 可視化與輸出**：

   * 顯示最終整合後 3D 路徑圖
   * 匯出 `uav_path.txt`，每行為一個 3D 座標點 `x y z`

---

## 輸出檔案說明

* `uav_path.txt`：最終 UAV 飛行路徑座標序列（Z 軸從 base 起跳，層與層間間隔固定）
* `slice_*.pcd`：每層切分後的點雲（會存入 `obstacle_slice_dir` 與 `shooting_slice_dir`）
* `Open3D` 視窗：會彈出視覺化畫面，包含每層與最終 3D 飛行路徑

---

## 相依套件

```bash
pip install open3d opencv-python tqdm scikit-learn networkx numpy
```
