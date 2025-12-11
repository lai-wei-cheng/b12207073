# 情緒辨識模型 (7-class Emotion Recognition)

## 模型概述
- **架構**: MobileNetV2 Transfer Learning
- **類別**: neutral, happy, sad, angry, disgust, fear, surprise
- **驗證準確度**: 60.91%
- **加權 F1 分數**: 0.6029

## 特點
- ✓ 熵值過濾: 移除高標註衝突的樣本 (entropy threshold=1.8)
- ✓ 輕度過採樣: 對少數類別 (neutral, disgust, fear) 進行 2x 過採樣
- ✓ 高效推論: MobileNetV2 適合邊界/實時應用
- ✓ 增強效果: 旋轉、位移、縮放等資料增強

## 使用方式

### 單張圖片預測
```python
from inference import load_trained_model, predict_emotion

model = load_trained_model()
result = predict_emotion("path/to/image.jpg", model)
print(f"情緒: {result['emotion']}, 信心: {result['confidence']:.3f}")
```

### 批量預測
```python
from inference import batch_predict, load_trained_model

model = load_trained_model()
results = batch_predict("path/to/image/folder", model)
```

### 命令行使用
```bash
# 單張圖片
python inference.py path/to/image.jpg

# 資料夾
python inference.py path/to/folder/
```

## 模型性能

### 整體指標
- Accuracy: 60.91%
- Macro F1: 0.4807
- Weighted F1: 0.6029

### 類別性能 (precision / recall / f1)
- **neutral**: 0.500 / 0.143 / 0.222 (樣本少，困難)
- **happy**: 0.857 / 0.923 / 0.889 (優秀)
- **sad**: 0.640 / 0.348 / 0.451 (中等)
- **angry**: 0.437 / 0.792 / 0.563 (高召回，低精度)
- **disgust**: 0.923 / 0.333 / 0.490 (高精度，低召回)
- **fear**: 0.000 / 0.000 / 0.000 (樣本極少，無法學習)
- **surprise**: 0.650 / 0.886 / 0.750 (強)

## 已知限制
1. **Fear**: 訓練樣本太少 (10 張)，模型無法學習 - 建議單獨收集 fear 樣本或用專門分類器
2. **Neutral**: 樣本和標註困難 - recall 只有 14.3%
3. **Imbalanced dataset**: 即使過採樣，仍存在類別不平衡

## 改進方向
- 收集更多 fear 和 neutral 樣本
- 嘗試更大的模型 (ResNet50) 但需更多計算資源
- 使用 SMOTE 或更激進的過採樣
- 考慮層級分類器 (先分 happy vs others，再細分)

## 訓練細節
- 資料集大小: 1,096 張 (熵值過濾後)
- 訓練集: 876, 驗證集: 220
- 優化器: Adam (lr=1e-3 head phase, 1e-5 finetune)
- Loss: Categorical Crossentropy
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

## 許可
MIT License - 自由使用和修改
