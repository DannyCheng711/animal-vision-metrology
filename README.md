# 動物影像分割、眼睛影像分割與距離量測

本專案是關於將圖像分割 (Image Segmentation) 模型的輸出應用於實際測量學 (Metrology) 任務的題目。

## 1. 專案介紹

本專案完成下列工作：

### 1. 資料集選取：

- 使用 COCO val2017，篩選出至少包含 2 隻非 crowd 狗（Dog）的圖片，共取得 29 張圖片。
- 由於 COCO 不提供狗眼睛的 Ground Truth，另使用 CVAT 人工標註眼睛位置與輪廓。

### 2. 動物輪廓 / 眼睛輪廓圖像分割：

- 使用 Yolo-seg models 進行動物輪廓圖像分割。
- 使用 HRNet 預測的眼睛 Keypoints，並透過 幾何橢圓 與 SAM 進行眼睛圖像輪廓分割。

### 3. 同一隻動物雙眼距離量測：

- 使用 HRNet 預測的左右眼 Keypoints 座標，計算同一隻狗雙眼之間的 Euclidean Distance，單位為 pixel。

### 4. 不同動物右眼距離量測：

- 使用 HRNet 預測的右眼 Keypoints 座標，計算不同隻狗右眼之間的 Euclidean Distance，單位為 pixel。

### 5. AI 模型選擇與評估：

- 動物影像分割：比較不同版本的 YOLO-seg 模型在本專案 COCO val2017 子集（至少 2 隻狗）上的表現，評估指標為 Mask AP。
- 眼睛 Keypoints 定位：比較 HRNet-w48（Heatmap）與 RTMPose-m（SimCC）在 CVAT 人工標註資料上的表現，評估指標為 Mean Keypoint Error 與 Coverage。
- 眼睛影像分割：比較 幾何橢圓 與 SAM 的眼睛分割結果，評估指標為 Mean IoU 與 Coverage。

### 6. 量測方法與驗證：

- 利用 Euclidean Distance 公式。並以 CVAT 人工標註作為 Ground Truth，評估模型預測距離與人工標註距離之間的誤差。

模型 Inference 結果：

- 請參考 [Inference 說明](docs/inference.md)。

AI 模型選擇與評估報告：

- 動物影像分割：[模型比較與 COCOeval 結果](docs/evaluation/animal_segmentation.md)
- 眼睛定位：[GT／YOLO bbox 定位誤差與 Coverage](docs/evaluation/eye_localization.md)
- 眼睛影像分割：[Geometric／SAM 的 IoU 與 Coverage](docs/evaluation/eye_segmentation.md)
- 眼睛距離量測：[同狗雙眼與跨狗右眼的 MAE／Relative Error](docs/evaluation/eye_distance.md)

## 2. 系統流程圖

目前由 `run/inference.py` 串接單張 COCO 圖片的推論、預測座標檢查、距離量測與結果輸出。

```text
COCO val2017 圖片（原圖）
│
▼
YOLO26x-seg：狗的圖像分割
├── 輸出 動物輪廓
└── 輸出 動物 Bounding Box
    │
    ▼
    HRNet-w48（pretrained on AP-10K）：左右眼定位
    │
    └── 左右眼 Keypoints 與 Confidence（原圖座標）
        │
        ├── SAM ViT-B：眼睛影像分割
        │   └── 輸出 左右眼 輪廓
        │
        └── 距離量測
            ├── 檢查預測眼睛是否有效 (座標位於原圖內)
            ├── 同一隻狗：左眼 ↔ 右眼
            └── 不同隻狗：右眼 ↔ 右眼（兩兩配對）
                └── Euclidean Distance（pixel）→ 距離圖與 JSON 輸出
```

## 3. 工作資料夾

```text
computer-vision/
├── README.md                          # 專案介紹與文件入口
├── dataset/                           # 資料集讀取、篩選與標註解析
│   ├── dataloader.py                  # 讀取 COCO 圖片與 instance annotations
│   ├── prepro.py                      # 資料篩選與選取清單匯出
│   └── eye_annotations.py             # 解析 CVAT 眼睛輪廓、點位與可見性標註
├── model/                             # 模型下載、載入與推論封裝
│   ├── __init__.py
│   ├── animal_segmentation.py         # YOLO-seg 動物實例分割
│   ├── eye_localization.py            # HRNet／RTMPose 眼睛 keypoint 定位
│   └── eye_segmentation.py            # Geometric／SAM 眼睛分割
├── measurement.py                     # Euclidean Distance 與 polygon mask 工具
├── run/                               # 推論與各任務評估入口
│   ├── inference.py                   # 單張圖片推論、視覺化與距離輸出
│   ├── animal_segmentation_evaluation.py
│   ├── eye_localization_evaluation.py
│   ├── eye_segmentation_evaluation.py
│   └── eye_distance_evaluation.py
├── docs/                              # 環境、資料集、模型與操作說明
│   ├── environment_setup.md
│   ├── dataset_setup.md
│   ├── model.md
│   ├── inference.md
│   └── evaluation/                    # 四項任務的評估方法與結果
├── data/                              # 本機資料集與人工標註
│   ├── coco/                          # COCO val2017 圖片與原始標註
│   └── cvat/dogs_min2/                # 篩選圖片與 CVAT 眼睛標註
├── weights/                           # 模型權重與 MMPose 設定檔
├── outputs/                           # 自動產生的結果
│   ├── selection/                     # 資料集選取清單
│   ├── inference/                     # 單張推論、mask、距離圖與 JSON
│   ├── animal_segmentation/           # 動物分割評估
│   ├── eye_localization/              # 眼睛定位評估
│   ├── eye_segmentation/              # 眼睛分割評估
│   └── eye_distance/                  # 眼睛距離評估
├── pyproject.toml                     # Python 版本、套件依賴與 uv 設定
├── uv.lock                            # 套件版本鎖定
├── .python-version                    # 專案 Python 版本
└── .gitignore                         # Git 忽略規則
```

`dataset/` 負責準備資料，`model/` 提供各階段模型，`run/` 串接推論或評估，`measurement.py` 提供共用量測工具。`data/`、`weights/` 與 `outputs/` 為本機資料與產出，已由 `.gitignore` 排除。

## 4. 環境建置

- [環境安裝](docs/environment_setup.md)
- [資料集準備](docs/dataset_setup.md)
- [模型設定與說明](docs/model.md)
