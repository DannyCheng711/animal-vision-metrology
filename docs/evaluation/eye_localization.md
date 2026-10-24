# 眼睛定位

固定第一階段選出的 YOLO26x-seg，以預測的動物 bbox 進行眼睛 keypoint 定位，作為後續眼睛分割與距離量測的輸入。COCO 標註未提供狗的眼睛 keypoint，因此另外使用 CVAT 人工標註。

## 人工標註資料

在原始圖片尺寸標註眼睛位置與區域，左右眼以動物自身方向為準。同張圖片以 `dog_id` 區分不同個體，並配對同一隻狗的 point 與 polygon。

| 標籤 | 類型 | 用途 |
|---|---|---|
| `L_Eye`、`R_Eye` | Point | 人工眼睛位置，作為 keypoint 評估參考 |
| `L_Eye_Mask`、`R_Eye_Mask` | Polygon | 眼睛區域，作為 segmentation 評估參考 |


| 情況 | Visibility | Measurable |
|---|---|---|
| 眼睛清楚可見，或部分可見但仍能判定位置 | `visible` | `yes` |
| 被頭部或毛髮遮擋，只能推測位置 | `uncertain` | `no` |


僅納入 `visibility=visible` 且 `measurable=yes` 的眼睛。CVAT 的 `dog_id` 透過 COCO segmentation 對應原始 annotation ID。

## 模型選用

使用 AP-10K 動物姿態資料集預訓練的模型，取得狗的左右眼位置。

| 模型 | Keypoint 表示方式 |
|---|---|
| HRNet-w48 | Heatmap：預測各 keypoint 的空間位置 |
| RTMPose-m | SimCC：將水平與垂直座標分別視為分類問題 |

## 評估設定

分兩部分比較：

- 先使用 COCO GT bbox 觀察定位模型的誤差
- 再使用 YOLO26x-seg 預測 bbox，觀察上游偵測的影響

兩組使用相同的 20 張有效標註圖片，共 62 個參考眼睛。

| 項目 | 設定 |
|---|---|
| YOLO Confidence | 0.5 |
| bbox 與 GT 配對 | Greedy 一對一配對，bbox IoU ≥ 0.50 |
| 有效 keypoint | 座標有限且位於原圖範圍內，不設 keypoint confidence 門檻 |

## 評估結果

### 第一部分：GT bbox

| 模型 | Mean Keypoint Error（px）↓ | 有效眼睛數／參考眼睛數 | Coverage ↑ |
|---|---:|---:|---:|
| **HRNet-w48** | **5.75** | 62／62 | 100% |
| RTMPose-m | 8.81 | 62／62 | 100% |

### 第二部分：YOLO26x-seg bbox

| 模型 | Mean Keypoint Error（px）↓ | 有效眼睛數／參考眼睛數 | Coverage ↑ | 
|---|---:|---:|---:|
| **HRNet-w48** | **8.65** | 53／62 | 85.5% | 
| RTMPose-m | 10.50 | 53／62 | 85.5% | 


- 每眼定位誤差（Keypoint Error）：

$$

e_i = \sqrt{(x_i^{pred} - x_i^{gt})^2 + (y_i^{pred} - y_i^{gt})^2}

$$

- Mean Keypoint Error：所有有效預測眼睛的定位誤差總和 / 有效預測眼睛數

- Coverage：有效預測眼睛數 / 全部參考眼睛數

未配對或無效座標不納入平均誤差，但保留於 Coverage 分母。


使用 YOLO26x-seg 預測 bbox 後，兩模型的 Coverage 均降至 85.5%，平均定位誤差也有所增加。由於兩組有效樣本數不同，再限定共同成功的 53 個眼睛進行比較：

| 模型 | GT Mean Error（px）↓ | YOLO Mean Error（px）↓ |
|---|---:|---:|
| HRNet-w48 | 3.59 | 8.65 |
| RTMPose-m | 5.07 | 10.50 |

相同樣本下的誤差增加，顯示上游 bbox 偏差會影響眼睛定位。HRNet-w48 在兩種 bbox 輸入下均取得較低的平均誤差，因此選為後續眼睛分割與距離量測的定位模型。

## 執行方式

從專案根目錄執行：

```bash
uv run python -m run.eye_localization_evaluation
```

程式依序評估 HRNet-w48、RTMPose-m 的 GT bbox 與 YOLO bbox 兩組結果。

| 輸出 | 位置 |
|---|---|
| GT bbox 逐眼結果與摘要 | `outputs/localization/{model}/ground_truth/` |
| YOLO bbox 逐眼結果與摘要 | `outputs/localization/{model}/yolo26x/` |
| 模型比較與 Coverage 降幅 | `outputs/localization/comparison.csv` |

各模型資料夾包含 `eye_errors.csv` 與 `summary.json`；`{model}` 為 `hrnet-w48` 或 `rtmpose-m`。
