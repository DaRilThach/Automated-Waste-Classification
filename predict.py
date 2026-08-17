import os
import sys
import argparse
import time
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

# Thiet lap UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import modules noi bo
sys.path.append(str(Path(__file__).resolve().parent))
from src.config import (
    CLASSES,
    CLASS_METADATA,
    CHECKPOINT_DIR,
    IMG_SIZE
)

class WastePredictor:
    def __init__(self, model_type="mobilenet_v2"):
        self.model_type = model_type
        checkpoint_path = CHECKPOINT_DIR / f"{model_type}_best.keras"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Khong tim thay checkpoint mo hinh tai: {checkpoint_path}")
        
        print(f"[*] Dang nap mo hinh {model_type}...")
        self.model = tf.keras.models.load_model(str(checkpoint_path))
        # Warmup
        dummy_input = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        self.model.predict(dummy_input, verbose=0)
        print("[+] Mo hinh da san sang suy luan!")

    def preprocess(self, image_input):
        """Tien xu ly anh PIL hoac numpy array thanh Tensor dau vao."""
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            img = Image.fromarray(image_input).convert("RGB")

        img_resized = img.resize(IMG_SIZE)
        img_array = np.array(img_resized, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch

    def predict(self, image_input, top_k=3):
        """Du doan phan loai rac thai cho 1 anh."""
        tensor = self.preprocess(image_input)
        
        start_time = time.perf_counter()
        predictions = self.model.predict(tensor, verbose=0)[0]
        latency_ms = (time.perf_counter() - start_time) * 1000

        top_indices = np.argsort(predictions)[::-1][:top_k]
        top_results = []
        for idx in top_indices:
            cls_name = CLASSES[idx]
            meta = CLASS_METADATA[cls_name]
            top_results.append({
                "class": cls_name,
                "vn_name": meta["vn_name"],
                "confidence": float(predictions[idx]),
                "confidence_percent": f"{predictions[idx]*100:.2f}%",
                "category": meta["category"],
                "is_recyclable": meta["is_recyclable"],
                "guide": meta["guide"]
            })

        best_cls = CLASSES[top_indices[0]]
        best_meta = CLASS_METADATA[best_cls]

        return {
            "predicted_class": best_cls,
            "vn_name": best_meta["vn_name"],
            "confidence": float(predictions[top_indices[0]]),
            "confidence_percent": f"{predictions[top_indices[0]]*100:.2f}%",
            "category": best_meta["category"],
            "is_recyclable": best_meta["is_recyclable"],
            "guide": best_meta["guide"],
            "latency_ms": round(latency_ms, 2),
            "top_k": top_results
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Du doan phan loai rac thai tu anh")
    parser.add_argument("--image", type=str, required=True, help="Duong dan den file anh can du doan")
    parser.add_argument("--model", type=str, default="mobilenet_v2", choices=["mobilenet_v2", "resnet50"], help="Loai mo hinh")
    args = parser.parse_args()

    predictor = WastePredictor(model_type=args.model)
    res = predictor.predict(args.image)

    print("\n" + "=" * 60)
    print("🔎 KET QUA PHAN LOAI RAC THAI (WASTE CLASSIFICATION RESULT)")
    print("=" * 60)
    print(f"Ảnh kiểm tra      : {args.image}")
    print(f"Nhãn dự đoán      : {res['predicted_class']} ({res['vn_name']})")
    print(f"Độ tự tin         : {res['confidence_percent']}")
    print(f"Phân loại tái chế : {res['category']}")
    print(f"Thời gian xử lý   : {res['latency_ms']} ms")
    print(f"Hướng dẫn xử lý   : {res['guide']}")
    print("-" * 60)
    print("Top 3 xác suất cao nhất:")
    for i, r in enumerate(res['top_k'], 1):
        print(f"  {i}. {r['class']} ({r['vn_name']}): {r['confidence_percent']}")
    print("=" * 60)
