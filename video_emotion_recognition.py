"""
視頻情緒辨識腳本
使用生產模式對視頻進行逐幀人臉情緒分析
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from collections import Counter
import os

# 情緒類別
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'disgust', 'fear', 'surprise']
EMOTION_LABELS_ZH = ['中性', '開心', '悲傷', '生氣', '厭惡', '恐懼', '驚訝']

def load_emotion_model():
    """載入生產模式"""
    model_path = 'production_model/model.h5'
    if not os.path.exists(model_path):
        model_path = 'Taiwanese/production_model/model.h5'
    if not os.path.exists(model_path):
        model_path = 'Taiwanese/fine_tuned_emotion_hard_entropy18_oversample.h5'
    
    print(f"載入模型: {model_path}")
    model = load_model(model_path)
    return model

def detect_faces(frame):
    """使用 Haar Cascade 偵測人臉"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def preprocess_face(face_img):
    """預處理人臉圖像以符合模型輸入"""
    face_resized = cv2.resize(face_img, (224, 224))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_normalized = face_rgb / 255.0
    face_batch = np.expand_dims(face_normalized, axis=0)
    return face_batch

def predict_emotion(model, face_img):
    """預測人臉情緒"""
    processed_face = preprocess_face(face_img)
    predictions = model.predict(processed_face, verbose=0)[0]
    emotion_idx = np.argmax(predictions)
    confidence = predictions[emotion_idx]
    return emotion_idx, confidence, predictions

def process_video(video_path, output_path=None, sample_rate=30):
    """
    處理視頻並進行情緒辨識
    
    參數:
    - video_path: 視頻檔案路徑
    - output_path: 輸出視頻路徑 (可選，None 則不保存)
    - sample_rate: 取樣率 (每 N 幀分析一次，降低計算量)
    """
    # 載入模型
    model = load_emotion_model()
    
    # 開啟視頻
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤: 無法開啟視頻 {video_path}")
        return None
    
    # 取得視頻屬性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\n視頻資訊:")
    print(f"  解析度: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  總幀數: {total_frames}")
    print(f"  時長: {duration:.2f} 秒")
    print(f"  取樣率: 每 {sample_rate} 幀分析一次\n")
    
    # 準備輸出視頻
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 紀錄所有情緒結果
    all_emotions = []
    frame_emotions = []
    
    frame_count = 0
    processed_count = 0
    
    print("開始處理視頻...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 按取樣率處理幀
        if frame_count % sample_rate == 0:
            # 偵測人臉
            faces = detect_faces(frame)
            
            frame_result = {
                'frame': frame_count,
                'timestamp': frame_count / fps if fps > 0 else 0,
                'num_faces': len(faces),
                'emotions': []
            }
            
            # 對每個人臉進行情緒辨識
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                emotion_idx, confidence, all_probs = predict_emotion(model, face_img)
                emotion_name = EMOTION_LABELS[emotion_idx]
                emotion_zh = EMOTION_LABELS_ZH[emotion_idx]
                
                all_emotions.append(emotion_name)
                frame_result['emotions'].append({
                    'emotion': emotion_name,
                    'emotion_zh': emotion_zh,
                    'confidence': float(confidence),
                    'bbox': (int(x), int(y), int(w), int(h))
                })
                
                # 在幀上標註
                color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                label = f"{emotion_zh} {confidence*100:.1f}%"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y-label_size[1]-10), (x+label_size[0], y), color, -1)
                cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            frame_emotions.append(frame_result)
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"已處理 {processed_count} 個取樣幀 ({frame_count}/{total_frames} 總幀)")
        
        # 寫入輸出視頻
        if out is not None:
            out.write(frame)
    
    # 釋放資源
    cap.release()
    if out is not None:
        out.release()
    
    print(f"\n視頻處理完成!")
    print(f"總共處理了 {processed_count} 個取樣幀 (共 {frame_count} 幀)")
    
    # 統計整體情緒分布
    if all_emotions:
        emotion_counts = Counter(all_emotions)
        print("\n整體情緒分布:")
        for emotion in EMOTION_LABELS:
            count = emotion_counts.get(emotion, 0)
            percentage = (count / len(all_emotions)) * 100 if len(all_emotions) > 0 else 0
            emotion_zh = EMOTION_LABELS_ZH[EMOTION_LABELS.index(emotion)]
            print(f"  {emotion_zh:6s} ({emotion:8s}): {count:4d} 次 ({percentage:5.1f}%)")
        
        # 找出主要情緒
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        dominant_emotion_zh = EMOTION_LABELS_ZH[EMOTION_LABELS.index(dominant_emotion)]
        print(f"\n主要情緒: {dominant_emotion_zh} ({dominant_emotion})")
    
    # 保存詳細結果到 CSV
    if frame_emotions:
        results_data = []
        for frame_result in frame_emotions:
            if frame_result['emotions']:
                for emotion_data in frame_result['emotions']:
                    results_data.append({
                        '幀數': frame_result['frame'],
                        '時間戳(秒)': round(frame_result['timestamp'], 2),
                        '情緒': emotion_data['emotion_zh'],
                        '英文': emotion_data['emotion'],
                        '信心度': round(emotion_data['confidence'], 3),
                        'X': emotion_data['bbox'][0],
                        'Y': emotion_data['bbox'][1],
                        '寬': emotion_data['bbox'][2],
                        '高': emotion_data['bbox'][3]
                    })
            else:
                results_data.append({
                    '幀數': frame_result['frame'],
                    '時間戳(秒)': round(frame_result['timestamp'], 2),
                    '情緒': '未偵測到人臉',
                    '英文': 'no_face',
                    '信心度': 0,
                    'X': 0, 'Y': 0, '寬': 0, '高': 0
                })
        
        df = pd.DataFrame(results_data)
        csv_path = video_path.rsplit('.', 1)[0] + '_emotions.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n詳細結果已保存到: {csv_path}")
    
    return {
        'total_frames': frame_count,
        'processed_frames': processed_count,
        'frame_emotions': frame_emotions,
        'emotion_distribution': dict(emotion_counts) if all_emotions else {},
        'dominant_emotion': dominant_emotion if all_emotions else None
    }

if __name__ == "__main__":
    # 視頻檔案路徑
    video_path = "vlog.mp4"
    
    # 輸出視頻路徑 (帶標註)
    output_path = "vlog_annotated.mp4"
    
    # 取樣率 (每 30 幀分析一次，約 1 秒分析一次，以 30fps 為例)
    sample_rate = 30
    
    # 處理視頻
    results = process_video(video_path, output_path=output_path, sample_rate=sample_rate)
    
    if results:
        print("\n處理完成!")
        print(f"標註後的視頻已保存到: {output_path}")
