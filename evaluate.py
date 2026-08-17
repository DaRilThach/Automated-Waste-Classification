import os
import sys
import argparse
import numpy as np
import tensorflow as tf
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
    OUTPUT_DIR,
    BATCH_SIZE
)
from src.dataset import load_datasets
from src.utils import (
    plot_confusion_matrix_custom,
    evaluate_predictions
)

def evaluate_model_pipeline(model_type="mobilenet_v2"):
    """
    Danh gia mo hinh tren tap kiem tra doc lap (Independent Test Set).
    """
    print("\n" + "=" * 70)
    print(f"📊 DANH GIA MO HINH TREN TAP TEST DOC LAP: {model_type.upper()}")
    print("=" * 70)

    checkpoint_path = CHECKPOINT_DIR / f"{model_type}_best.keras"
    if not checkpoint_path.exists():
        print(f"❌ Khong tim thay file checkpoint tai: {checkpoint_path}")
        print("💡 Vui long chay 'python train.py --model " + model_type + "' truoc.")
        return None

    # 1. Nap tap Test
    print("[*] Dang nap tap Test...")
    _, _, test_ds, _ = load_datasets(batch_size=BATCH_SIZE)

    # 2. Nap Model
    print(f"[*] Dang nap mo hinh tu {checkpoint_path}...")
    model = tf.keras.models.load_model(str(checkpoint_path))

    # 3. Danh gia tong quan qua model.evaluate
    eval_results = model.evaluate(test_ds, verbose=1)
    loss = eval_results[0]
    accuracy = eval_results[1]
    top2_accuracy = eval_results[2] if len(eval_results) > 2 else None

    print("\n" + "-" * 50)
    print(f"📈 KET QUA DANH GIA TONG THE ({model_type.upper()}):")
    print(f"   - Test Loss          : {loss:.4f}")
    print(f"   - Test Accuracy      : {accuracy * 100:.2f}%")
    if top2_accuracy:
        print(f"   - Test Top-2 Accuracy: {top2_accuracy * 100:.2f}%")
    print("-" * 50)

    # 4. Lay toan bo Ground Truth va Du doan xac suat
    y_true_list = []
    y_pred_probs_list = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_pred_probs_list.append(probs)
        y_true_list.append(labels.numpy())

    y_true = np.argmax(np.concatenate(y_true_list, axis=0), axis=-1)
    y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)

    # 5. Tinh toan cac chi so Precision, Recall, F1-Score
    report, y_pred = evaluate_predictions(
        y_true=y_true,
        y_pred_probs=y_pred_probs,
        classes=CLASSES,
        model_name=model_type.upper()
    )

    # 6. Ve va luu Ma tran nham lan (Confusion Matrix)
    cm_path = OUTPUT_DIR / f"{model_type}_confusion_matrix.png"
    plot_confusion_matrix_custom(
        y_true=y_true,
        y_pred=y_pred,
        classes=CLASSES,
        model_name=model_type.upper(),
        save_path=cm_path
    )

    # 7. In bang chi tiet ket qua
    print("\n" + "=" * 70)
    print(f"📋 BANG CHI TIET DANG GIA THEO TUNG LOP ({model_type.upper()}):")
    print("-" * 70)
    print(f"{'Lop Rac':<12} | {'Tieng Viet':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'So mau'}")
    print("-" * 70)
    for cls, metrics in report["per_class"].items():
        print(f"{cls:<12} | {metrics['vn_name']:<15} | {metrics['precision']*100:>8.2f}% | "
              f"{metrics['recall']*100:>8.2f}% | {metrics['f1_score']*100:>8.2f}% | {metrics['support']:>6}")
    print("-" * 70)
    print(f"{'Macro Avg':<29} | {report['macro_avg']['precision']*100:>8.2f}% | "
          f"{report['macro_avg']['recall']*100:>8.2f}% | {report['macro_avg']['f1_score']*100:>8.2f}% | {len(y_true):>6}")
    print(f"{'Weighted Avg':<29} | {report['weighted_avg']['precision']*100:>8.2f}% | "
          f"{report['weighted_avg']['recall']*100:>8.2f}% | {report['weighted_avg']['f1_score']*100:>8.2f}% | {len(y_true):>6}")
    print("=" * 70)

    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Danh gia mo hinh phan loai rac thai tren tap Test")
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_v2",
        choices=["mobilenet_v2", "resnet50", "all"],
        help="Chon mo hinh can danh gia"
    )
    args = parser.parse_args()

    if args.model == "all":
        evaluate_model_pipeline("mobilenet_v2")
        evaluate_model_pipeline("resnet50")
    else:
        evaluate_model_pipeline(args.model)
