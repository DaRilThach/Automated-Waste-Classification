import os
import sys
import argparse
import tensorflow as tf
from pathlib import Path
import json

# Thiet lap ma hoa Unicode cho console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import modules noi bo
sys.path.append(str(Path(__file__).resolve().parent))
from src.config import (
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    PHASE_1_EPOCHS,
    PHASE_1_LR,
    PHASE_2_EPOCHS,
    PHASE_2_LR,
    BATCH_SIZE
)
from src.dataset import load_datasets
from src.models import (
    build_mobilenet_v2,
    build_resnet50,
    unfreeze_model_for_finetuning
)
from src.utils import plot_training_history

def train_model_pipeline(model_type="mobilenet_v2", epochs_phase1=PHASE_1_EPOCHS, epochs_phase2=PHASE_2_EPOCHS):
    """
    Quy trinh Huan luyen toan dien 2 giai doan (Two-Stage Transfer Learning & Fine-Tuning).
    """
    print("\n" + "=" * 70)
    print(f"🚀 KHOI DONG HUAN LUYEN MO HINH: {model_type.upper()}")
    print("=" * 70)

    # 1. Nap du lieu
    print("[*] Dang nap du lieu Train, Validation va Test...")
    train_ds, val_ds, test_ds, class_weights = load_datasets(batch_size=BATCH_SIZE)
    print(f"[+] Class Weights da tinh: {class_weights}")

    # 2. Xay dung mo hinh
    if model_type == "mobilenet_v2":
        model, base_model = build_mobilenet_v2(freeze_base=True)
        checkpoint_path = CHECKPOINT_DIR / "mobilenet_v2_best.keras"
        csv_log_path = OUTPUT_DIR / "mobilenet_v2_training.csv"
        unfreeze_layers = -30
    elif model_type == "resnet50":
        model, base_model = build_resnet50(freeze_base=True)
        checkpoint_path = CHECKPOINT_DIR / "resnet50_best.keras"
        csv_log_path = OUTPUT_DIR / "resnet50_training.csv"
        unfreeze_layers = -25
    else:
        raise ValueError(f"Khong ho tro loai mo hinh: {model_type}")

    model.summary()

    # =========================================================================
    # GIAI DOAN 1: FEATURE EXTRACTION (Base Model dong bang)
    # =========================================================================
    print("\n" + "-" * 70)
    print(f"📌 GIAI DOAN 1: WARMUP & FEATURE EXTRACTION ({epochs_phase1} Epochs, LR={PHASE_1_LR})")
    print("-" * 70)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE_1_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy")]
    )

    callbacks_phase1 = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(csv_log_path), append=False)
    ]

    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase1,
        class_weight=class_weights,
        callbacks=callbacks_phase1,
        verbose=1
    )

    # =========================================================================
    # GIAI DOAN 2: FINE-TUNING (Mo dong bang cac layer cuoi voi LR thap)
    # =========================================================================
    if epochs_phase2 > 0:
        print("\n" + "-" * 70)
        print(f"📌 GIAI DOAN 2: FINE-TUNING ({epochs_phase2} Epochs, LR={PHASE_2_LR})")
        print("-" * 70)

        unfreeze_model_for_finetuning(base_model, unfreeze_from_layer=unfreeze_layers)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE_2_LR),
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy")]
        )

        callbacks_phase2 = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=2,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(str(csv_log_path), append=True)
        ]

        total_epochs = len(history_p1.history['accuracy']) + epochs_phase2
        history_p2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=total_epochs,
            initial_epoch=len(history_p1.history['accuracy']),
            class_weight=class_weights,
            callbacks=callbacks_phase2,
            verbose=1
        )

        # Gop lich su huan luyen
        full_history = {}
        for k in history_p1.history:
            full_history[k] = history_p1.history[k] + history_p2.history[k]
    else:
        full_history = history_p1.history

    # 3. Ve va luu bieu do training
    plot_training_history(
        full_history,
        model_name=model_type.upper(),
        save_path=OUTPUT_DIR / f"{model_type}_training_history.png"
    )

    print(f"\n[+] Huan luyen {model_type.upper()} hoan tat thanh cong!")
    print(f"[+] Trong so tot nhat duoc luu tai: {checkpoint_path}")
    return model, full_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huan luyen mo hinh phan loai rac thai")
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_v2",
        choices=["mobilenet_v2", "resnet50", "all"],
        help="Chon mo hinh huan luyen (mobilenet_v2, resnet50, hoac all)"
    )
    parser.add_argument("--epochs1", type=int, default=PHASE_1_EPOCHS, help="So epoch giai doan 1")
    parser.add_argument("--epochs2", type=int, default=PHASE_2_EPOCHS, help="So epoch giai doan 2")

    args = parser.parse_args()

    if args.model == "all":
        train_model_pipeline("mobilenet_v2", args.epochs1, args.epochs2)
        train_model_pipeline("resnet50", args.epochs1, args.epochs2)
    else:
        train_model_pipeline(args.model, args.epochs1, args.epochs2)
