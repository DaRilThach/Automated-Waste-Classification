import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from PIL import Image
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, CLASS_METADATA, OUTPUT_DIR

def plot_training_history(history_dict, model_name="MobileNetV2", save_path=None):
    """
    Ve bieu do qua trinh huan luyen: Loss va Accuracy theo tung Epoch.
    """
    acc = history_dict.get('accuracy', [])
    val_acc = history_dict.get('val_accuracy', [])
    loss = history_dict.get('loss', [])
    val_loss = history_dict.get('val_loss', [])
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Accuracy Plot
    ax1.plot(epochs_range, acc, 'o-', color='#2980b9', linewidth=2, label='Training Accuracy')
    ax1.plot(epochs_range, val_acc, 's-', color='#27ae60', linewidth=2, label='Validation Accuracy')
    ax1.set_title(f'{model_name} - Accuracy theo Epoch', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', frameon=True)
    ax1.set_ylim([0.0, 1.05])
    
    # 2. Loss Plot
    ax2.plot(epochs_range, loss, 'o-', color='#e74c3c', linewidth=2, label='Training Loss')
    ax2.plot(epochs_range, val_loss, 's-', color='#e67e22', linewidth=2, label='Validation Loss')
    ax2.set_title(f'{model_name} - Loss theo Epoch', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    
    if save_path is None:
        save_path = OUTPUT_DIR / f"{model_name.lower()}_training_history.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Da luu bieu do huan luyen tai: {save_path}")

def plot_confusion_matrix_custom(y_true, y_pred, classes=CLASSES, model_name="MobileNetV2", save_path=None):
    """
    Ve va luu Ma tran nham lan (Confusion Matrix) chi tiet voi ca gia tri thuc te va ty le phan tram.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Nhãn hiển thị kèm tên tiếng Việt
    display_labels = [f"{cls}\n({CLASS_METADATA[cls]['vn_name']})" for cls in classes]

    plt.figure(figsize=(9, 7))
    sns.set_theme(style="white")
    
    # Tạo text hiển thị cả số lượng tuyệt đối và phần trăm
    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_matrix[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm_norm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar=True,
        linewidths=1.0,
        linecolor="#bdc3c7"
    )

    plt.title(f"Confusion Matrix (Ma trận nhầm lẫn) - {model_name}", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Nhãn Dự Đoán (Predicted Label)", fontsize=12, labelpad=10)
    plt.ylabel("Nhãn Thực Tế (Ground Truth Label)", fontsize=12, labelpad=10)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    if save_path is None:
        save_path = OUTPUT_DIR / f"{model_name.lower()}_confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Da luu ma tran nham lan tai: {save_path}")

def evaluate_predictions(y_true, y_pred_probs, classes=CLASSES, model_name="MobileNetV2"):
    """
    Tinh toan toan bo cac chi so danh gia hoc thong ke:
    - Overall Accuracy
    - Macro / Weighted Precision, Recall, F1-Score
    - Per-class Precision, Recall, F1-Score, Support
    """
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    # Chi tiet tung lop
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    per_class_metrics = {}
    for idx, cls in enumerate(classes):
        per_class_metrics[cls] = {
            "vn_name": CLASS_METADATA[cls]["vn_name"],
            "category": CLASS_METADATA[cls]["category"],
            "precision": float(p_class[idx]),
            "recall": float(r_class[idx]),
            "f1_score": float(f1_class[idx]),
            "support": int(support_class[idx])
        }

    report = {
        "model_name": model_name,
        "accuracy": acc,
        "macro_avg": {
            "precision": float(p_macro),
            "recall": float(r_macro),
            "f1_score": float(f1_macro)
        },
        "weighted_avg": {
            "precision": float(p_weighted),
            "recall": float(r_weighted),
            "f1_score": float(f1_weighted)
        },
        "per_class": per_class_metrics
    }

    # Luu ket qua danh gia ra JSON
    json_path = OUTPUT_DIR / f"{model_name.lower()}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"[+] Da luu chi tiet danh gia vao: {json_path}")

    return report, y_pred
