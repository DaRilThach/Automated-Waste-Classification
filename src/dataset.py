import tensorflow as tf
from pathlib import Path
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    CLASSES,
    IMG_SIZE,
    BATCH_SIZE,
    RANDOM_SEED
)

def get_data_augmentation():
    """
    Xay dung pipeline Data Augmentation tang cuong du lieu cho tap Train.
    Bao gom lat anh, xoay goc ngau nhien, zoom, dich chuyen va dieu chinh tuong phan.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=RANDOM_SEED),
        tf.keras.layers.RandomRotation(0.15, fill_mode="reflect", seed=RANDOM_SEED),
        tf.keras.layers.RandomZoom(0.15, fill_mode="reflect", seed=RANDOM_SEED),
        tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=RANDOM_SEED),
        tf.keras.layers.RandomContrast(0.1, seed=RANDOM_SEED),
    ], name="data_augmentation")

def load_datasets(batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    """
    Nap tap Train, Val, Test bang tf.keras.utils.image_dataset_from_directory
    va toi uu hoa luong du lieu bang cache va prefetch (AUTOTUNE).
    """
    # 1. Nap tap Train
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=RANDOM_SEED
    )

    # 2. Nap tap Validation
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False
    )

    # 3. Nap tap Test
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False
    )

    # Tinh toan Class Weights de xu ly mat can bang lop (VD: lop trash it mau hon cac lop khac)
    labels = []
    for _, y in train_ds:
        labels.extend(np.argmax(y.numpy(), axis=-1))
    
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=np.array(labels)
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}

    # Toi uu hieu nang voi AUTOTUNE
    AUTOTUNE = tf.data.AUTOTUNE
    data_aug = get_data_augmentation()

    # Áp dụng Augmentation cho Train
    train_ds_perf = train_ds.map(
        lambda x, y: (data_aug(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    ).prefetch(buffer_size=AUTOTUNE)

    val_ds_perf = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds_perf = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds_perf, val_ds_perf, test_ds_perf, class_weight_dict
