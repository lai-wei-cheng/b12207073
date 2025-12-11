# 作業完成報告

## 作業需求 vs 檔案映射

| 需求 | 相關檔案 |
|------|----------|
| **需求 1**: Taiwanese 人臉驗證 | Taiwanese/faces_256x256/ 資料集 (1,232 張) + esults/confusion_face256_hard.csv |
| **需求 2**: 視頻驗證工具 | ideo_emotion_recognition.py |
| **需求 3**: vlog.mp4 分析結果 | log_emotions.csv (逐幀分析) + log_annotated.mp4 (標註視頻) |
| **需求 4**: GitHub 提交 | https://github.com/lai-wei-cheng/b12207073.git |

## 最佳模型性能

- **架構**: MobileNetV2 (轉移學習)
- **準確率**: 63.27%
- **加權 F1**: 0.6029
- **訓練資料**: 1,096 張 (熵值 1.8 過濾 + 2 倍過採樣)
- **驗證資料**: 245 張

## 核心檔案

```
production_model/                      # 部署模型
 model.h5                          # MobileNetV2 權重 (27.7 MB)
 metadata.json                     # 訓練配置和性能
 inference.py                      # 推論腳本
 README_zh.md                      # 使用文檔

Taiwanese/                             # 資料集和結果
 faces_256x256/                    # 1,232 張原始圖像
 Image_info.xls                    # 標籤和熵值
 results/
    confusion_face256_hard.csv    # 需求 1: 驗證結果
    model_comparison.csv          # 模型對比
    architecture_comparison.csv   # 架構對比
 video_emotion_recognition.py      # 需求 2: 視頻分析
 vlog_emotions.csv                 # 需求 3: 逐幀結果
 vlog_annotated.mp4                # 需求 3: 標註視頻
```

## vlog.mp4 分析結果

- 主要情緒: Happy 64.7% (33 幀)
- 次要情緒: Surprise 23.5% (12 幀)
- 分析幀數: 50 幀 (50.13 秒視頻)

## 關鍵策略

- **熵值過濾**: 使用 Image_info.xls 中的 EntropyVal，移除高熵值 (> 1.8) 的模糊樣本
- **過採樣**: 2 倍複製少數類別 (neutral, disgust, fear)
- **結果**: 比基線提升 F1 分數 (0.5946  0.6029)
