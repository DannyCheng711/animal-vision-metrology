# 動物影像分割、眼睛影像分割與距離量測

使用 COCO val2017 中至少有兩隻非 crowd 狗的 29 張圖片，依序比較動物分割、眼睛 mask 與 keypoint 距離。眼睛參考標註來自 CVAT；僅納入符合有效標註條件的眼睛。


## 1. 動物影像分割

比較 COCO 預訓練的 YOLOv8x-seg、YOLO11x-seg 與 YOLO26x-seg，選擇後續流程的上游模型。目前不設部署資源限制，以分割品質為優先。

> 選用原因：YOLO-seg 可同時輸出動物 bbox 與 instance mask，並提供 COCO 預訓練權重，方便銜接眼睛定位。YOLO12 尚無官方預訓練 segmentation 權重，因此不納入比較。

### 評估設定

三個模型使用相同的 29 張圖片，以原始 COCO instance mask 為參考，透過 `pycocotools.COCOeval` 進行 segmentation 評估，僅評估 dog 類別。

| 適用模型 | 輸入尺寸設定 | 評估 Confidence | IoU 門檻 | 裝置 |
|---|---|---|---|---|
| YOLO-seg x | imgsz=640（等比例縮放與 padding） | 0.001 | 0.50–0.95（間隔 0.05） | NVIDIA H100 80GB HBM3 |

### 評估結果

| 模型 | Mask AP@[.50:.95] ↑ | AP50 ↑ | AP75 ↑ |
|---|---:|---:|---:|
| YOLOv8x-seg | 0.551 | 0.725 | 0.670 |
| YOLO11x-seg | 0.556 | 0.718 | 0.671 |
| **YOLO26x-seg** | **0.595** | **0.791** | **0.706** |

- Mask AP@[.50:.95]：在 0.50～0.95、間隔 0.05 的 IoU 門檻下，計算 AP 後取平均。
- AP50／AP75：分別為 IoU 門檻 0.50／0.75 下的 AP，依 confidence 排序累積 Precision–Recall 曲線。

YOLO26x-seg 在本次比較的三項 AP 指標均取得最高分，因此選為後續眼睛定位、眼睛影像分割與距離量測的上游模型。

### 執行方式

從專案根目錄執行：

```bash
uv run python -m model.animal_segmentation
```

程式依序評估三個模型。各模型輸出位於 `outputs/animal_segmentation/{model}/coco_eval/`，包含 COCO 格式預測 `coco_predictions.json`、評估摘要 `summary.json`、二值 mask 與疊圖；比較總表為 `outputs/animal_segmentation/comparison_coco.csv`。

## 2. 眼睛 Mask：Geometric 與 SAM 增強比較

固定第一階段的 YOLO 與 HRNet AP-10K，使用 YOLO 預測 bbox 取得眼睛 keypoint。兩種方法共用相同 keypoint，與 CVAT 眼睛 polygon 比較。

### 眼睛定位模型比較（規劃中）

先比較定位模型，再固定一個模型產生眼睛 mask 的提示點。以下欄位暫定，模型權重與結果於實驗後補上；目前 localization 已整合 HRNet 與 RTMPose-m AP-10K（`architecture="rtmpose-m"`），ViTPose 待實作；完整比較數字待填。

| Model | Representation | Mean Keypoint Error（px）↓ | PCK@0.05 ↑ | Keypoint Coverage ↑ | Latency（ms／圖）↓ |
|---|---|---|---|---|---|
| HRNet | Heatmap | 待填 | 待填 | 待填 | 待填 |
| ViTPose | Heatmap | 待填 | 待填 | 待填 | 待填 |
| RTMPose | SimCC | 待填 | 待填 | 待填 | 待填 |

- 使用具動物眼睛關鍵點定義的權重，統一左右眼順序、YOLO bbox、有效人工標註與評估樣本。
- Mean Keypoint Error：成功輸出之有效預測點與人工 point 的平均歐氏距離；須搭配 Coverage 解讀。
- PCK@0.05：誤差不超過參考尺度 5% 的眼睛比例。正規化尺度待決定，實驗前固定；缺失預測視為不正確，分母為所有可評估眼睛。
- Keypoint Coverage：有有效預測座標的眼睛數／所有可評估眼睛數，包含上游漏偵測造成的缺失；confidence 門檻待固定。此欄與下方 Mask Coverage 分開計算。
- Latency：相同裝置、batch size 與暖機條件下的每圖定位推論時間；輸入尺寸及是否包含前後處理待固定，不含模型載入。

### 人工標註方式

使用 CVAT 在原圖尺寸標註，左右眼以狗自身方向為準。同張圖片內以 `dog_id` 區分不同狗，並將同一隻狗的 point 與 polygon 配對。

| 標籤 | 類型 | 用途 |
|---|---|---|
| `L_Eye`、`R_Eye` | 單點 | 人工眼睛位置，作為 keypoint 距離的參考 |
| `L_Eye_Mask`、`R_Eye_Mask` | Polygon | 人工眼睛區域，作為 mask IoU 的參考 |

僅納入 `visibility=visible` 且 `measurable=yes` 的 point；`uncertain` 或 `measurable=no` 的推測位置不納入評估。眼睛 mask 必須對應同一隻狗、同側的有效 point。CVAT 的 `dog_id` 為圖片內編號，透過 COCO 狗的 segmentation 對應原始 annotation ID。

### 評估方式：IoU 與 Coverage

分割品質僅使用 IoU：它直接表示預測區域與人工標註的重疊程度，便於解釋 geometric baseline 與 SAM 增強的差異，不另列 Dice。

**未成功產生有效 mask 的眼睛也必須納入評估，不能只計算成功樣本。** 所有符合人工標註條件且具有有效參考 polygon 的眼睛都列入分母；即使上游 YOLO 漏偵測，也不能從分母移除。

- `Coverage = 成功產生有效 mask 的眼睛數 / 所有可評估眼睛數`。
- 漏偵測、無效提示或空 mask：該眼睛 IoU 設為 0，並計入缺失／失敗數。
- `Mean IoU = 所有可評估眼睛的 IoU 總和 / 所有可評估眼睛數`，包含失敗樣本的 0 分。
- 有效輸出須為對應眼睛的原圖尺寸、非空二值 mask，並通過事先固定的輸出檢查；這不代表輪廓正確，品質仍由 IoU 衡量。額外的合理性條件若有使用，須在實驗前固定並註明，不能看結果後挑樣本。

例如 20 個可評估眼睛中，12 個有有效 mask、其平均 IoU 為 0.75，另外 8 個失敗，則 Coverage 為 60%，整體 Mean IoU 為 `12 × 0.75 / 20 = 0.45`。

| 方法 | 作法 | Mean IoU（含失敗）↑ | Coverage ↑ | 可評估眼睛數 | 有效 Mask 數 | 缺失／失敗數 |
|---|---|---|---|---|---|---|
| Geometric baseline | 由 keypoint 與估計半徑產生橢圓 mask | 待填 | 待填 | 待填 | 待填 | 待填 |
| Geometric + SAM | Point + 幾何 box 提示 SAM，再做空間過濾 | 待填 | 待填 | 待填 | 待填 | 待填 |

> 本節為完整流程的評估規範；目前使用 COCO bbox 的眼睛評估結果，尚不能代表包含 YOLO 漏偵測的 Coverage。後續程式與實驗須依上述規則更新。

| 固定設定 | 數值 |
|---|---|
| YOLO 權重 | 待填 |
| HRNet 權重 | 待填 |
| SAM 權重 | 待填 |
| 幾何半徑比例與上下限 | 待填 |
| 有效眼睛標註規則 | 待填 |

觀察與失敗案例：待填。

## 3. Keypoint 距離量測

沿用前兩階段的預測 keypoint，與 CVAT 人工 point 計算的距離比較。距離不使用 mask 中心，因此兩種眼睛 mask 方法共用一份距離結果。不同狗的右眼只在同張圖片內建立不重複配對。

| 量測項目 | 有效參考筆數 | 成功量測筆數 | 未完成筆數 | MAE（px）↓ | MSE（px²）↓ |
|---|---|---|---|---|---|
| 同一隻狗的左右眼距離 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 不同狗的右眼距離 | 待填 | 待填 | 待填 | 待填 | 待填 |

| 指標 | 公式 |
|---|---|
| Mask IoU | `交集面積 / 聯集面積` |
| 歐氏距離 | `d = sqrt((x₁ − x₂)² + (y₁ − y₂)²)` |
| MAE | `mean(abs(d_pred − d_GT))` |
| MSE | `mean((d_pred − d_GT)²)` |

距離使用原圖座標，單位為像素，不代表實際物理距離。MAE 為主要結果，MSE 補充呈現較大誤差；未完成量測需另外記錄，不能視為零誤差。

結論與限制：待填。
