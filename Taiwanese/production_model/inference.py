"""
Production inference script for 7-class emotion recognition model.
"""
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = Path(__file__).parent / "model.h5"
CLASSES = ["neutral", "happy", "sad", "angry", "disgust", "fear", "surprise"]
IMG_SIZE = 224

def load_trained_model():
    """Load the pre-trained emotion model."""
    return load_model(str(MODEL_PATH))

def predict_emotion(image_path, model=None):
    """
    Predict emotion from a single image.
    
    Args:
        image_path (str or Path): Path to image file
        model: Loaded model (optional, will load if None)
    
    Returns:
        dict: {
            "emotion": str (predicted class),
            "confidence": float (probability),
            "all_probs": dict (all class probabilities)
        }
    """
    if model is None:
        model = load_trained_model()
    
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_batch, verbose=0)[0]
    
    # Get results
    pred_idx = np.argmax(preds)
    pred_emotion = CLASSES[pred_idx]
    confidence = float(preds[pred_idx])
    
    all_probs = {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}
    
    return {
        "emotion": pred_emotion,
        "confidence": confidence,
        "all_probs": all_probs,
    }

def batch_predict(image_dir, model=None):
    """Predict emotions for all images in a directory."""
    if model is None:
        model = load_trained_model()
    
    results = {}
    img_dir = Path(image_dir)
    for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
        try:
            result = predict_emotion(img_path, model)
            results[img_path.name] = result
        except Exception as e:
            results[img_path.name] = {"error": str(e)}
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path_or_dir>")
        sys.exit(1)
    
    model = load_trained_model()
    path = Path(sys.argv[1])
    
    if path.is_dir():
        results = batch_predict(path, model)
        for fname, result in results.items():
            if "error" in result:
                print(f"{fname}: ERROR - {result['error']}")
            else:
                print(f"{fname}: {result['emotion']} ({result['confidence']:.3f})")
    else:
        result = predict_emotion(path, model)
        print(f"Emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"All probabilities: {result['all_probs']}")
