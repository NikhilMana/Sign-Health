import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ─── Configuration ────────────────────────────────────────────────────────────
MIN_SAMPLES    = 1    # Include every class with at least 1 sample
SEQ_CAP        = 60   # Hard cap on sequence length (frames)
AUGMENT_TARGET = 20   # Boost small classes to at least this many samples
NOISE_STD      = 0.005
TIME_SHIFT     = 3    # Max frames to shift left/right during augmentation
# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    mp_data_path = Path("MP_DATA")
    sequences, labels = [], []
    for class_dir in mp_data_path.iterdir():
        if class_dir.is_dir():
            files = list(class_dir.glob("*.npy"))
            if len(files) >= MIN_SAMPLES:
                for npy_file in files:
                    sequences.append(np.load(npy_file).astype(np.float32))
                    labels.append(class_dir.name)
    return sequences, labels


def augment_data(sequences, labels):
    """Targeted augmentation: small classes get boosted to AUGMENT_TARGET,
    all classes always get 2x augments for additional variance."""
    class_seqs = defaultdict(list)
    for seq, lab in zip(sequences, labels):
        class_seqs[lab].append(seq)

    aug_seq  = list(sequences)
    aug_labels = list(labels)

    for label, seqs in class_seqs.items():
        n = len(seqs)
        needed   = max(0, AUGMENT_TARGET - n)   # bring up small classes
        extra    = n * 2                          # always 2x baseline augment
        total_aug = max(needed, extra)

        for i in range(total_aug):
            base = seqs[i % n]
            noise     = np.random.normal(0, NOISE_STD, base.shape).astype(np.float32)
            shift     = np.random.randint(-TIME_SHIFT, TIME_SHIFT + 1)
            augmented = np.roll(base + noise, shift, axis=0)
            scale     = np.random.uniform(0.95, 1.05)
            augmented = (augmented * scale).astype(np.float32)
            aug_seq.append(augmented)
            aug_labels.append(label)

    print(f"  Original: {len(sequences)} | After augmentation: {len(aug_seq)}")
    return aug_seq, aug_labels


def preprocess_data(sequences, labels):
    actual_max = max(len(s) for s in sequences)
    max_length = min(actual_max, SEQ_CAP)
    print(f"  Sequence max_length: {max_length} (data max was {actual_max})")

    X = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_length, padding='post', truncating='post', dtype='float32'
    )

    label_encoder = LabelEncoder()
    y_encoded     = label_encoder.fit_transform(labels)
    y_categorical = keras.utils.to_categorical(y_encoded)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
        )
    except ValueError:
        print("  [WARN] Stratified split failed (some classes too small). Using random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42
        )

    return X_train, X_test, y_train, y_test, label_encoder, max_length, y_encoded


def create_model(input_shape, num_classes):
    """Larger capacity Bi-LSTM for 272-class vocabulary."""
    model = keras.Sequential([
        layers.Bidirectional(layers.LSTM(128, return_sequences=True), input_shape=input_shape),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(256, return_sequences=False)),
        layers.Dropout(0.25),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train():
    print("="*70)
    print("TRAINING — FINE-TUNED FOR NEW 272-CLASS DATASET")
    print("="*70)

    sequences, labels = load_data()
    class_dist = Counter(labels)
    print(f"Loaded: {len(sequences)} samples across {len(class_dist)} classes")
    print(f"  Min/class: {min(class_dist.values())} | Max: {max(class_dist.values())} | Avg: {np.mean(list(class_dist.values())):.1f}")

    sequences, labels = augment_data(sequences, labels)

    X_train, X_test, y_train, y_test, label_encoder, max_length, y_encoded_all = preprocess_data(sequences, labels)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded_all), y=y_encoded_all)
    class_weight_dict = dict(enumerate(class_weights))

    print(f"\n✓ Class weights | range: {min(class_weights):.2f} – {max(class_weights):.2f}")
    print(f"✓ Train: {len(X_train)} | Val: {len(X_test)} | Classes: {len(label_encoder.classes_)}")

    model = create_model((max_length, X_train.shape[2]), len(label_encoder.classes_))
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint('models/best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    batch_size = 32  # smaller batch for better gradient updates across many small classes

    print(f"\nTraining: batch={batch_size}, max_epochs=250, early_stop patience=20")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=250,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*70}")
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"{'='*70}")

    models_dir = Path("models")
    model.save(models_dir / "sign_model.keras")

    import pickle
    with open(models_dir / "label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)
    with open(models_dir / "max_length.txt", 'w') as f:
        f.write(str(max_length))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history['loss'],     label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Loss Curve'); ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(True)

    ax2.plot(history.history['accuracy'],     label='Train Acc')
    ax2.plot(history.history['val_accuracy'], label='Val Acc')
    ax2.set_title('Accuracy Curve'); ax2.set_xlabel('Epoch'); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(models_dir / "training_history.png", dpi=300)
    print(f"\n✅ Model saved → models/sign_model.keras")
    print(f"✅ Plot saved  → models/training_history.png")


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    train()
