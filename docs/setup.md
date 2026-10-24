# 安裝與執行

環境需求：

- Ubuntu 22.04／24.04
- Linux x86_64
- Python 3.11

套件版本由 `pyproject.toml` 與 `uv.lock` 固定。

GPU 執行需要主機已安裝可支援 CUDA 11.8 runtime 的 NVIDIA driver。Python 套件則是使用 PyTorch CUDA 11.8 與對應的預編譯 MMCV wheel；一般不需另外安裝完整 CUDA Toolkit。

無 GPU 時程式會改用 CPU，但 SAM 推論與完整評估會較慢。

## 1. 下載專案與系統依賴

以下指令在遠端 server 的 Bash 終端執行。若已下載專案，直接進入包含 `pyproject.toml` 的根目錄。

```bash
# 更新系統套件索引
sudo apt-get update

# 安裝下載與解壓工具、編譯工具及 OpenCV 所需的系統函式庫
sudo apt-get install -y git curl unzip build-essential libgl1 libglib2.0-0

git clone https://github.com/DannyCheng711/animal-vision-metrology-.git

cd animal-vision-metrology-
```

沒有 sudo 權限時，請管理員安裝上述系統套件；若 server 已具備這些套件，可略過安裝步驟。

## 2. 建立虛擬環境

安裝與目前開發環境一致的 uv 0.13.1；已安裝相同版本時可略過。安裝方式參考 [uv 官方文件](https://docs.astral.sh/uv/getting-started/installation/)。

```bash
# 安裝指定版本的 uv
curl -LsSf https://astral.sh/uv/0.13.1/install.sh | sh

# 將 uv 的預設安裝目錄加入目前終端的 PATH
export PATH="$HOME/.local/bin:$PATH"

# 確認 uv 安裝成功與版本
uv --version

# 建立 .venv、依鎖檔安裝套件，並執行環境檢查
bash scripts/setup.sh

# 在目前終端啟用虛擬環境，後續 python 指令將使用此環境
source .venv/bin/activate
```

`setup.sh` 會使用 Python 3.11 建立 `.venv`，依 `uv.lock` 安裝套件並檢查相依性，再測試模型套件匯入、MMCV CPU 運算子，以及 torchvision CPU／可用 GPU 的運算子。

看到 `Environment check passed` 表示套件及運算子檢查完成。GPU server 也應顯示 `CUDA available: True`。若顯示 `False`，程式將使用 CPU。

每次重新連線，只需進入專案根目錄並啟用環境：

```bash
source .venv/bin/activate
python -m run.check_environment
```

## 3. 驗證單張圖片推論

先下載一張 COCO 範例圖片，不需要 COCO annotations 或 CVAT 標註：

```bash
# 建立 COCO 驗證集圖片目錄
mkdir -p data/coco/val2017

# 下載單張範例圖片；失敗最多重試 3 次
curl -fL --retry 3 \
  -o data/coco/val2017/000000081766.jpg \
  https://s3.amazonaws.com/images.cocodataset.org/val2017/000000081766.jpg

# 執行動物與眼睛分割、距離量測，結果輸出至 outputs/inference/000000081766/
python -m run.inference --image-id 000000081766
```

首次執行會自動下載 YOLO26x-seg、HRNet-w48 的設定與權重，以及 SAM ViT-B 權重至 `weights/`。需要網路存取 GitHub、OpenMMLab 與 Meta 的模型下載站；之後重用本機檔案。模型下載與推論可能需要數分鐘。

成功後檢查 `outputs/inference/000000081766/`，應包含 `overlay.png`、兩張距離圖、`predictions.json`、`distances.json` 與 `masks/`。此圖目前範例偵測到 3 隻狗；距離為原圖 pixel。範例圖片與結果見 [Inference 說明](inference.md)。

## 4. 準備 COCO 資料集

完整評估需要 COCO val2017 圖片及原始 instance annotations。以下檔案總下載量約 1 GB，解壓後也需預留數 GB 空間。

```bash
# 建立 COCO 資料目錄
mkdir -p data/coco

# 下載驗證集圖片；失敗最多重試 3 次，支援續傳
curl -fL --retry 3 -C - \
  -o data/coco/val2017.zip \
  https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip

# 下載 COCO 2017 訓練集與驗證集標註壓縮檔
curl -fL --retry 3 -C - \
  -o data/coco/annotations_trainval2017.zip \
  https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip

# 解壓圖片至 data/coco/val2017/；同名檔案會覆寫
unzip -oq data/coco/val2017.zip -d data/coco

# 僅解壓驗證集的 instance annotations
unzip -oq data/coco/annotations_trainval2017.zip \
  annotations/instances_val2017.json -d data/coco

# 篩選至少包含 2 隻非 crowd 狗的圖片，匯出選取清單
python -m dataset.prepro
```

`dataset.prepro` 會篩選至少包含 2 隻非 crowd 狗的圖片，預期取得 29 張，並輸出 `outputs/selection/two_or_more_dogs.csv`。保留原始 COCO 標註，不需修改或另做圖片裁切。

## 5. CVAT 標註與評估

眼睛定位、眼睛分割與距離評估使用本專案的 CVAT 人工標註，已隨 repo 提供，clone 後不需另外下載或解壓：

```text
annotations/coco_dogs_eyes.zip
```

ZIP 僅含根目錄的 `annotations.xml`（CVAT for images XML 格式），不含圖片，標註中的圖片名稱對應 COCO 原圖檔名。標籤、`dog_id`、`visibility` 與 `measurable` 規則見[眼睛定位評估](evaluation/eye_localization.md)。

從專案根目錄依序執行：

```bash
python -m run.animal_segmentation_evaluation
python -m run.eye_localization_evaluation
python -m run.eye_segmentation_evaluation
python -m run.eye_distance_evaluation
```

所有結果寫入 `outputs/`，對應設定與結果見 [README 的四份評估報告入口](../README.md#1-專案介紹)。
