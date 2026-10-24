# 動物影像分割

比較 COCO 預訓練的 YOLO-seg 模型，選擇後續流程的上游模型。目前不設部署資源限制，以分割品質為優先。

## 資料選用

使用 COCO val2017 中至少有兩隻非 crowd 狗的 29 張圖片。結果僅代表此子集，不代表完整 COCO val2017 的成績。

## 模型選用

YOLO-seg 可同時輸出動物 bbox 與 輪廓 (masks)。並提供 COCO 預訓練權重，方便銜接眼睛定位。依 [Ultralytics YOLO12 官方文件](https://docs.ultralytics.com/models/yolo12/)，YOLO12 未提供官方預訓練 segmentation 權重，因此不納入比較。

## 評估設定

三個模型使用相同的 29 張圖片，以原始 COCO instance mask 為參考，透過 `pycocotools.COCOeval` 進行 segmentation 評估，僅評估 dog 類別。

| 適用模型 | 輸入尺寸設定 | 評估 Confidence | IoU 門檻 |
|---|---|---|---|
| YOLO-seg x | imgsz=640（等比例縮放與 padding） | 0.001 | 0.50–0.95（間隔 0.05） |

評估使用較低的 confidence 門檻 0.001，以保留候選預測供 AP 的 Precision–Recall 計算。COCOeval 採預設 `maxDets=[1, 10, 100]`，下表中的 AP 使用每張圖最多 100 個預測。圖片篩選時只計非 crowd 狗；評分時保留原始標註，由 COCOeval 處理 crowd 區域。

## 評估結果

| 模型 | Mask AP@[.50:.95] ↑ | AP50 ↑ | AP75 ↑ |
|---|---:|---:|---:|
| YOLOv8x-seg | 0.551 | 0.725 | 0.670 |
| YOLO11x-seg | 0.556 | 0.718 | 0.671 |
| **YOLO26x-seg** | **0.595** | **0.791** | **0.706** |

- Mask AP@[.50:.95]：在 0.50～0.95、間隔 0.05 的 IoU 門檻下，計算 AP 後取平均。
- AP50／AP75：分別為 IoU 門檻 0.50／0.75 下的 AP，依 confidence 排序累積 Precision–Recall 曲線。

YOLO26x-seg 在本次比較的三項 AP 指標均取得最高分，因此選為後續眼睛定位、眼睛影像分割與距離量測的上游模型。

## 執行方式

從專案根目錄執行：

```bash
uv run python -m run.animal_segmentation_evaluation
```

程式依序評估 YOLOv8x-seg、YOLO11x-seg 與 YOLO26x-seg。

## 輸出

以下為 animal_segmentation evaluation 輸出的結果

| 輸出 | 位置 |
|---|---|
| COCO 格式 mask RLE 與 confidence | `outputs/animal_segmentation/{model}/coco_eval/coco_predictions.json` |
| 評估摘要（AP／AR） | `outputs/animal_segmentation/{model}/coco_eval/summary.json` |
| 二值 mask | `outputs/animal_segmentation/{model}/coco_eval/masks/` |
| 分割疊圖與 bbox | `outputs/animal_segmentation/{model}/coco_eval/overlays/` |
| 三個模型比較總表 | `outputs/animal_segmentation/comparison_coco.csv` |

`{model}` 為 `yolov8x`、`yolo11x` 或 `yolo26x`。
