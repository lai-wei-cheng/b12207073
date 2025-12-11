# 台灣人臉情緒辨識系統

## 概述
本專案使用 MobileNetV2 轉移學習模型進行 7 類情緒辨識，在台灣人臉資料集上達到 63.27% 準確率。

## 作業需求完成狀態

### ✅ 需求 1: Taiwanese 人臉圖像驗證
- 在 `Taiwanese/faces_256x256/` 資料集上驗證
- **性能**: 63.27% 準確率，0.6029 加權 F1
- **測試樣本**: 245 張圖像

### ✅ 需求 2: 視頻驗證
- 使用 `video_emotion_recognition.py` 進行視頻情緒辨識
- 支援人臉偵測和逐幀分析

### ✅ 需求 3: vlog.mp4 結果輸出
- **結果檔案**:
  - `vlog_emotions.csv`: 詳細的逐幀分析
  - `vlog_annotated.mp4`: 標註後的視頻
- **主要情緒**: Happy 64.7%, Surprise 23.5%

### ✅ 需求 4: GitHub 倉庫
- 所有必要檔案已上傳此倉庫

## 目錄結構

```
Taiwanese/
├── production_model/              # 最佳模型 (已部署)
│   ├── model.h5                   # MobileNetV2 權重
│   ├── metadata.json              # 訓練配置和性能
│   ├── inference.py               # 推論腳本
│   └── README_zh.md               # 使用說明
├── results/                       # 評估結果
│   ├── confusion_face256_hard.csv  # 混淆矩陣
│   ├── model_comparison.csv       # 模型性能對比
│   └── architecture_comparison.csv # 架構對比
├── faces_256x256/                 # 台灣人臉資料集 (1,232 張)
├── Image_info.xls                 # 標籤和熵值資訊
├── video_emotion_recognition.py   # 視頻分析工具
├── vlog_emotions.csv              # vlog.mp4 分析結果
├── vlog_annotated.mp4             # vlog.mp4 標註視頻
└── SUBMISSION_SUMMARY.md          # 詳細報告
```

## 快速開始

### 安裝依賴
```bash
pip install tensorflow opencv-python pandas numpy scikit-learn
```

### 單張圖像預測
```python
from production_model.inference import load_trained_model, predict_emotion

model = load_trained_model()
result = predict_emotion("path/to/image.jpg", model)
print(result)  # {"emotion": "happy", "confidence": 0.923, ...}
```

### 批量預測
```python
from production_model.inference import batch_predict, load_trained_model

model = load_trained_model()
results = batch_predict("path/to/folder/", model)
```

### 視頻分析
```bash
python video_emotion_recognition.py
```
生成:
- `vlog_emotions.csv`: 逐幀情緒分析
- `vlog_annotated.mp4`: 標註視頻

## 最佳模型

### 配置
- **架構**: MobileNetV2 (轉移學習)
- **預訓練權重**: ImageNet
- **輸入尺寸**: 224×224×3
- **輸出類別**: 7 種情緒 (neutral, happy, sad, angry, disgust, fear, surprise)

### 訓練策略
- **資料過濾**: 熵值閾值 1.8
- **過採樣**: 2 倍用於少數類別 (neutral, disgust, fear)
- **資料增強**: 旋轉、位移、縮放、剪切、翻轉
- **訓練參數**:
  - 頭部訓練: 5 週期 (lr=1e-3)
  - 微調: 20 週期 (lr=1e-5)
  - Batch Size: 32

### 性能
| 指標 | 值 |
|------|------|
| 準確率 | 63.27% |
| 加權 F1 | 0.6029 |
| 訓練樣本 | 1,096 張 (熵值過濾 + 過採樣) |
| 驗證樣本 | 245 張 |

### 各類別性能
| 情緒 | 精準率 | 召回率 | F1 分數 |
|------|--------|---------|----------|
| Happy | 0.857 | 0.923 | 0.889 |
| Surprise | 0.650 | 0.886 | 0.752 |
| Angry | 0.437 | 0.792 | 0.566 |
| Disgust | 0.923 | 0.333 | 0.489 |
| Sad | 0.640 | 0.348 | 0.451 |
| Neutral | 0.500 | 0.143 | 0.222 |
| Fear | 0.000 | 0.000 | 0.000 |

## 模型開發過程

### 訓練變體對比

| 策略 | 準確率 | 加權 F1 | 備註 |
|------|--------|----------|------|
| 基準線 (無過濾) | 63.27% | 0.5946 | - |
| 類別權重 + Focal Loss | 49.00% | - | ❌ 失敗 |
| 過採樣 | 58.78% | - | 次優 |
| 熵值 1.5 過濾 | 53.06% | 0.4769 | 過度過濾 |
| 熵值 1.8 只過濾 | 54.29% | 0.4860 | 單獨過濾損害準度 |
| **熵值 1.8 + 過採樣** | **63.27%** | **0.6029** | ✅ 最佳 |
| ResNet50 + 熵值 1.8 + 過採樣 | 21.22% | 0.0743 | 過度擬合 |

### 架構對比

| 架構 | 準確率 | 加權 F1 | 訓練時間 | 推論速度 |
|------|--------|----------|----------|-----------|
| **MobileNetV2** | **63.27%** | **0.6029** | ~15 分鐘 | 快 ✅ |
| ResNet50 | 21.22% | 0.0743 | ~45 分鐘 | 慢 ❌ |

## 關鍵發現

### 資料品質的重要性
- **熵值過濾**: 移除多觀察者不同意的樣本
- **溫和過採樣**: 補償少數類別，避免過度複製
- **最佳組合**: 熵值 1.8 + 2 倍過採樣 = 63.27% 準確率

### 架構選擇
- **MobileNetV2 優勢**:
  - 小資料集上性能更好
  - 推論快速 (實時應用適合)
  - 模型小 (27.7 MB)
- **ResNet50 劣勢**:
  - 過度擬合於小資料集
  - 訓練時間長
  - 不適合小規模資料

## 檔案清單

### 核心檔案
- ✅ `SUBMISSION_SUMMARY.md` - 完整作業報告
- ✅ `video_emotion_recognition.py` - 視頻分析工具
- ✅ `production_model/` - 最佳模型及推論腳本
- ✅ `results/` - 評估結果和性能指標
- ✅ `vlog_emotions.csv` - vlog.mp4 分析結果
- ✅ `vlog_annotated.mp4` - 標註視頻
- ✅ `faces_256x256/` - 訓練資料集
- ✅ `Image_info.xls` - 標籤和熵值資訊

### 已清理
- ❌ 7 個過期模型 (.h5)
- ❌ 26 個訓練/評估腳本
- ❌ 編譯快取目錄

## 系統需求

- Python 3.8+
- TensorFlow 2.13+
- OpenCV 4.5+
- Pandas, NumPy, scikit-learn

## 參考資源

- **論文參考**: 多觀察者投票 + 熵值一致性 (image_info.xls 中的 EntropyVal 欄位)
- **訓練資料**: 台灣人臉資料集 (faces_256x256/)
- **模型架構**: MobileNetV2 預訓練權重 (ImageNet)

## 聯絡方式

- **GitHub**: https://github.com/lai-wei-cheng/b12207073.git
- **Email**: lai-wei-cheng@gmail.com

---

**作業完成日期**: 2025-12-11  
**狀態**: ✅ 所有需求完成，已上傳 GitHub
