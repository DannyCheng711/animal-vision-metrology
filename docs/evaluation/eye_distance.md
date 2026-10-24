# 眼睛距離量測

串接 YOLO26x-seg → HRNet-w48，使用預測 keypoint 計算原圖像素距離，不使用眼睛 mask 中心。人工可見性沿用[眼睛定位評估](eye_localization.md)的有效標註規則。

- **Intra-animal Distance**：同一隻狗的左右眼皆有有效人工標註，且兩個預測點皆有效時，計算雙眼距離。
- **Inter-animal Right-eye Distance**：同張圖片內，不同狗的右眼皆有有效人工標註及預測時，計算所有不重複配對的距離。

## 評估設定

與 Inference 使用相同的模型封裝及 `measurement.point_distance`；YOLO detection confidence 為 0.5，預測狗與 COCO GT 以 bbox IoU ≥ 0.5 進行一對一配對。預測眼睛座標須為有限值且位於原圖內。

## 評估結果

| 量測項目 | 成功／參考筆數 | 量測 Coverage ↑ | MAE（px）↓ | Mean Relative Error ↓ |
|---|---:|---:|---:|---:|
| Intra-animal Distance | 22／26 | 84.6% | 3.88 | 13.96% |
| Inter-animal Right-eye Distance | 12／19 | 63.2% | 1.17 | 0.59% |


其中，量測 Coverage 定義為「成功量測配對數／全部有效人工標註配對數」：

- 兩個端點皆有有效人工標註，才列為參考配對。
- 參考配對的兩個預測端點皆有效，才算成功量測。

誤差僅統計成功量測的配對，失敗配對不視為零誤差。距離單位為 pixel，不代表實際物理距離。


相關公式如下：

- 兩點距離（Euclidean Distance）：

$$
d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
$$

- Mean Absolute Error（MAE）：

$$
\mathrm{MAE} = \frac{1}{N}\sum_{i=1}^{N}\left|d_i^{pred} - d_i^{gt}\right|
$$

- Mean Relative Error（%）：

$$
\mathrm{Mean\ Relative\ Error}\ (\%) = \frac{1}{N}\sum_{i=1}^{N}\frac{\left|d_i^{pred} - d_i^{gt}\right|}{d_i^{gt}} \times 100
$$

其中 $N$ 為各量測項目的成功量測筆數。



## 執行方式

從專案根目錄執行：

```bash
uv run python -m run.eye_distance_evaluation
```

| 輸出 | 位置 |
|---|---|
| 逐筆距離、誤差與狀態 | `outputs/eye_distance/yolo26x_hrnet-w48/distance_errors.csv` |
| 指標與推論設定 | `outputs/eye_distance/yolo26x_hrnet-w48/summary.json` |
