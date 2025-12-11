# 台灣人臉情緒辨識系統

## 作業需求 vs 檔案映射

| 需求 | 相關檔案 |
|------|----------|
| **需求 1**: Taiwanese 人臉驗證 | aces_256x256/ (資料集) + esults/confusion_face256_hard.csv (驗證結果：63.27%) |
| **需求 2**: 視頻驗證工具 | ideo_emotion_recognition.py |
| **需求 3**: vlog.mp4 結果 | log_emotions.csv + log_annotated.mp4 |
| **需求 4**: GitHub 提交 | 本倉庫 |

## 核心檔案結構

```
 production_model/              # 最佳模型
    model.h5                   # MobileNetV2 (27.7 MB)
    metadata.json              # 訓練配置
    inference.py               # 推論腳本
 faces_256x256/                 # 資料集 (1,232 張)
 Image_info.xls                 # 標籤和熵值
 results/
    confusion_face256_hard.csv  # 混淆矩陣
    model_comparison.csv       # 模型對比
    architecture_comparison.csv # 架構對比
 video_emotion_recognition.py   # 視頻分析
 vlog_emotions.csv              # vlog 逐幀結果
 vlog_annotated.mp4             # vlog 標註視頻
```

## 模型性能

- **準確率**: 63.27%
- **加權 F1**: 0.6029
- **架構**: MobileNetV2
- **訓練策略**: 熵值 1.8 過濾 + 2 倍過採樣

## 快速使用

單張圖像：
```python
from production_model.inference import load_trained_model, predict_emotion
model = load_trained_model()
result = predict_emotion("image.jpg", model)
```

視頻分析：
```bash
python video_emotion_recognition.py
```

## 系統需求

- Python 3.8+
- TensorFlow 2.13+, OpenCV 4.5+, Pandas, NumPy
