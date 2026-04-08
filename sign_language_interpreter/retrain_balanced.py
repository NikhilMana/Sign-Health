"""
Retrain with balanced class weights.

Uses the shared modules for constants and loads data from MP_DATA_QUALITY.
Leverages DeviceManager for GPU acceleration, mixed precision, and tf.data.
"""

import sys
import pickle
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.constants import (
    MAX_SEQ_LENGTH_CAP,
    DATA_AUGMENTATION_NOISE,
    DATA_AUGMENTATION_COUNT,
    ENABLE_MIXED_PRECISION,
    ENABLE_XLA,
    GPU_MEMORY_GROWTH,
    DEFAULT_BATCH_SIZE,
)
from shared.device_config import DeviceManager

# ── Hardware initialisation (runs once at import time) ────────
_device_mgr = DeviceManager(
    enable_mixed_precision=ENABLE_MIXED_PRECISION,
    enable_xla=ENABLE_XLA,
    enable_memory_growth=GPU_MEMORY_GROWTH,
)
_strategy = _device_mgr.initialize()


def load_data(data_dir="MP_DATA_QUALITY"):
    """Load keypoint sequences and labels from the data directory."""
    mp_data_path = Path(data_dir)
    sequences, labels = [], []

    for class_dir in mp_data_path.iterdir():
        if class_dir.is_dir():
            for npy_file in class_dir.glob("*.npy"):
                sequences.append(np.load(npy_file).astype(np.float32))
                labels.append(class_dir.name)

    return sequences, labels


def augment_data(sequences, labels):
    """Augment with Gaussian noise."""
    aug_seq, aug_labels = list(sequences), list(labels)
    for seq, label in zip(sequences, labels):
        for _ in range(DATA_AUGMENTATION_COUNT):
            noise = np.random.normal(0, DATA_AUGMENTATION_NOISE, seq.shape).astype(np.float32)
            aug_seq.append(seq + noise)
            aug_labels.append(label)
    return aug_seq, aug_labels


def preprocess_data(sequences, labels):
    """Pad, encode labels, and split into train/test."""
    max_length = min(max(len(s) for s in sequences), MAX_SEQ_LENGTH_CAP)
    X = keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_length, padding="post", truncating="post", dtype="float32"
    )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    y_categorical = keras.utils.to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
    )
    return X_train, X_test, y_train, y_test, label_encoder, max_length, y_encoded


def create_model(input_shape, num_classes):
    """Build a Bidirectional LSTM classifier.

    When mixed precision is active the softmax output layer stays float32
    for numerical stability.
    """
    output_dtype = "float32" if _device_mgr.is_mixed_precision_active else None

    return keras.Sequential([
        layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=input_shape),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(128, return_sequences=False)),
        layers.Dropout(0.4),
        layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax", dtype=output_dtype),
    ])


def evaluate_per_class(model, X_test, y_test, label_encoder, output_dir):
    """Generate per-class metrics and confusion matrix heatmap."""
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    report = classification_report(
        y_true_classes, y_pred_classes,
        target_names=label_encoder.classes_,
        output_dict=True,
    )

    with open(output_dir / "class_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    # Confusion matrix heatmap
    try:
        import seaborn as sns
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        n_classes = len(label_encoder.classes_)
        fig_size = max(10, n_classes * 0.6)
        plt.figure(figsize=(fig_size, fig_size * 0.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
        plt.close()
        print(f"✓ Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")
    except ImportError:
        print("⚠ seaborn not installed — skipping confusion matrix heatmap")

    # Flag weak classes
    weak = [cls for cls, m in report.items()
            if isinstance(m, dict) and m.get("f1-score", 1) < 0.7]
    if weak:
        print(f"\n⚠ Low-performing classes (F1 < 0.7): {weak}")

    return report


def train():
    """Main training loop with class weights.

    Model is built inside the distribution strategy scope for GPU / multi-GPU
    readiness, and training data is served via an optimised tf.data pipeline.
    """
    print("=" * 70)
    print("TRAINING WITH CLASS WEIGHTS (BALANCED)")
    print("=" * 70)

    sequences, labels = load_data()
    print(f"Loaded: {len(sequences)} samples, {len(set(labels))} classes")

    sequences, labels = augment_data(sequences, labels)
    print(f"After augmentation: {len(sequences)} samples")

    X_train, X_test, y_train, y_test, label_encoder, max_length, y_encoded = preprocess_data(
        sequences, labels
    )

    # Class weights
    class_weights = compute_class_weight("balanced", classes=np.unique(y_encoded), y=y_encoded)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\n✓ Class weight range: {min(class_weights):.2f} – {max(class_weights):.2f}")

    # ── Build model inside strategy scope for GPU / multi-GPU ─
    with _strategy.scope():
        model = create_model((max_length, X_train.shape[2]), len(label_encoder.classes_))
        model.compile(
            optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    models_dir = Path("models")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(str(models_dir / "best_model.keras"), monitor="val_accuracy", save_best_only=True),
    ]

    # ── Build tf.data pipelines with prefetching ─────────────
    batch_size = DEFAULT_BATCH_SIZE
    train_ds = _device_mgr.build_dataset(X_train, y_train, batch_size=batch_size, training=True)
    val_ds = _device_mgr.build_dataset(X_test, y_test, batch_size=batch_size, training=False)

    print(f"\n  Device  : {_device_mgr.device_tag}")
    print(f"  Batch   : {batch_size}")
    print(f"  Mixed FP: {_device_mgr.is_mixed_precision_active}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=150,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'=' * 70}")
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"{'=' * 70}")

    # GPU memory summary
    _device_mgr.print_memory_usage()

    # Save artifacts
    model.save(models_dir / "sign_model.keras")
    with open(models_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    with open(models_dir / "max_length.txt", "w") as f:
        f.write(str(max_length))

    # Per-class evaluation
    evaluate_per_class(model, X_test, y_test, label_encoder, models_dir)

    # Model registry
    try:
        from shared.model_registry import ModelRegistry
        registry = ModelRegistry(str(models_dir))
        registry.register_model(
            metrics={"accuracy": float(test_acc), "loss": float(test_loss)},
            description="Balanced retrain (GPU-accelerated)",
        )
    except Exception:
        pass  # Registry is optional

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["loss"], label="Train")
    ax1.plot(history.history["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history["accuracy"], label="Train")
    ax2.plot(history.history["val_accuracy"], label="Val")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(models_dir / "training_history.png", dpi=300)
    print(f"\n✅ Balanced model saved!")


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    train()
