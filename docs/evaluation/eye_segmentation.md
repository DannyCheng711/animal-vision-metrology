# 眼睛影像分割

固定 YOLO26x-seg 與 HRNet-w48，兩種方法共用相同的預測 bbox 與眼睛 keypoint。

| 方法 | 作法 |
|---|---|
| Geometric | 由 keypoint 與預測雙眼距離估計橢圓 mask |
| SAM ViT-B | Point + Box Prompt → Highest Predicted IoU Mask |

## 評估設定

使用 20 張圖片中具有有效 point 與 polygon 的 62 個眼睛。

幾何橢圓與 SAM Box Prompt 使用相同半徑比例 0.15，水平半徑限制為 2～20 px，垂直半徑為水平的 0.75 倍且至少 1 px。


YOLO detection confidence 為 0.5，預測 bbox 與 GT 以 bbox IoU ≥ 0.5 進行 Greedy 一對一配對。

評估公式如下：

- Mask IoU = 預測 mask 與人工 mask 的交集面積 / 聯集面積
- Mean IoU = 全部可評估眼睛的 IoU 總和 / 可評估眼睛數
- Coverage = 成功產生有效 mask 的眼睛數 / 可評估眼睛數


漏偵測、無效提示或空 mask 的 IoU 計 0。


## 評估結果

| 方法 | Mean IoU ↑ | Coverage ↑ | 有效 Mask／參考眼睛數 | 缺失／失敗數 |
|---|---:|---:|---:|---:|
| Geometric | 0.220 | 85.5% | 53／62 | 9 |
| SAM ViT-B | **0.342** | 85.5% | 53／62 | 9 |

SAM 的 Mean IoU 高於幾何橢圓，Coverage 相同；兩組均包含上游未配對造成的 9 個眼睛缺失。

## 執行方式

```bash
uv run python -m run.eye_segmentation_evaluation
```

| 輸出 | 位置 |
|---|---|
| Geometric 逐眼結果、摘要、mask 與疊圖 | `outputs/eye_segmentation/yolo26x_hrnet-w48/geometric/` |
| SAM ViT-B 直接輸出結果 | `outputs/eye_segmentation/yolo26x_hrnet-w48/sam/` |
| 比較總表 | `outputs/eye_segmentation/yolo26x_hrnet-w48/comparison.csv` |
