import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import IMG_HEIGHT, IMG_WIDTH, CHANNELS, NUM_CLASSES

def build_mobilenet_v2(num_classes=NUM_CLASSES, input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), freeze_base=True):
    """
    Xay dung mo hinh MobileNetV2 Transfer Learning:
    - Toi uu hoa cho suy luan thoi gian thuc (Real-time inference tren Web/Edge).
    - Su dung tien xu ly dac thu cua MobileNetV2 (chuan hoa [-1, 1]).
    - Head tuy bien voi BatchNormalization va Dropout de chong overfitting.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # Tien xu ly anh chuan hoa cho MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Base Model duoc pre-train tren ImageNet
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = not freeze_base
    
    # Trich xuat dac trung
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    
    # Classification Head
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_feature"
    )(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="classification_output"
    )(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="WasteClassifier_MobileNetV2")
    return model, base_model

def build_resnet50(num_classes=NUM_CLASSES, input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), freeze_base=True):
    """
    Xay dung mo hinh ResNet50 Transfer Learning:
    - Kien truc Residual Network 50 lop giup hoc cac bieu dien dac trung phuc tap va do sau cao.
    - Su dung tien xu ly dac thu cua ResNet50 (chuyen ve BGR va tru mean).
    - Head tuy bien voi 512 units va Regularization.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # Tien xu ly anh chuan hoa cho ResNet50
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    
    # Base Model pre-trained tren ImageNet
    base_model = tf.keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = not freeze_base
    
    # Trich xuat dac trung
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    
    # Classification Head
    x = layers.Dense(
        512,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_feature_1"
    )(x)
    x = layers.Dropout(0.5, name="dropout_1")(x)
    
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="classification_output"
    )(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="WasteClassifier_ResNet50")
    return model, base_model

def unfreeze_model_for_finetuning(base_model, unfreeze_from_layer=-30):
    """
    Mo dong bang (Unfreeze) cac layer cuoi cua Base Model de tien hanh Fine-tuning giai doan 2.
    """
    base_model.trainable = True
    
    # Giu dong bang cac layer dau, chi train cac layer sau unfreeze_from_layer
    for layer in base_model.layers[:unfreeze_from_layer]:
        layer.trainable = False
        
    print(f"[+] Da mo dong bang {abs(unfreeze_from_layer)} layers cuoi cua {base_model.name} de Fine-Tuning.")
