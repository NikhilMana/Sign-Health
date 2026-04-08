import os

import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers, regularizers

from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from pathlib import Path

import matplotlib.pyplot as plt

from collections import Counter

import sys

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.device_config import DeviceManager
from shared.constants import (
    ENABLE_MIXED_PRECISION,
    ENABLE_XLA,
    GPU_MEMORY_GROWTH,
    DEFAULT_BATCH_SIZE,
    MAX_SEQ_LENGTH_CAP,
)
from shared.augmentation import augment_dataset

# ── Hardware initialisation (runs once at import time) ────────
_device_mgr = DeviceManager(
    enable_mixed_precision=ENABLE_MIXED_PRECISION,
    enable_xla=ENABLE_XLA,
    enable_memory_growth=GPU_MEMORY_GROWTH,
)
_strategy = _device_mgr.initialize()

# Add this constant at the top of the file after imports

MIN_SAMPLES_PER_CLASS = 10



def load_sign_language_data():
    """
    Load sign language data from MP_DATA folder.

    Returns:
        sequences, labels, groups, sign_labels

    *groups* tracks the source video so train/test can be split
    without leaking data from the same video into both sets.
    """
    print("Loading sign language data...")

    mp_data_path = Path("MP_DATA")
    if not mp_data_path.exists():
        raise FileNotFoundError(
            "MP_DATA folder not found. Please run process_videos.py first."
        )

    sequences = []
    labels = []
    groups = []  # video-level group ID for GroupKFold / group-aware split

    sign_labels = sorted(d.name for d in mp_data_path.iterdir() if d.is_dir())
    print(f"Found {len(sign_labels)} sign labels")

    for sign_label in sign_labels:
        sign_folder = mp_data_path / sign_label
        npy_files = sorted(sign_folder.glob("*.npy"))
        print(f"  {sign_label}: {len(npy_files)} sequences")

        for npy_file in npy_files:
            sequence = np.load(npy_file).astype(np.float32)
            sequences.append(sequence)
            labels.append(sign_label)
            groups.append(f"{sign_label}/{npy_file.stem}")

    print(f"\nTotal sequences loaded: {len(sequences)}")
    if sequences:
        print(f"Sequence shape example: {sequences[0].shape}")

    # Count per class
    label_counts = Counter(labels)
    print("\nSamples per class:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    # Filter out rare classes
    filtered_seqs, filtered_labels, filtered_groups = [], [], []
    removed_classes = set()
    kept_classes = set()

    for seq, label, group in zip(sequences, labels, groups):
        if label_counts[label] >= MIN_SAMPLES_PER_CLASS:
            filtered_seqs.append(seq)
            filtered_labels.append(label)
            filtered_groups.append(group)
            kept_classes.add(label)
        else:
            removed_classes.add(label)

    if removed_classes:
        print(f"\nRemoved {len(removed_classes)} rare classes: {sorted(removed_classes)}")
    print(f"Kept {len(kept_classes)} classes, {len(filtered_seqs)} sequences")

    return filtered_seqs, filtered_labels, filtered_groups, sorted(kept_classes)



# augment_data() removed — replaced by shared.augmentation.augment_dataset()
# which is applied AFTER train/test split to avoid data leakage.



def preprocess_data(sequences, labels):

    """

    Preprocess the data for training:

    1. Pad or truncate sequences to a fixed length

    2. Convert labels to numerical format

    3. One-hot encode labels

    4. Split into train/test sets

    """

    print("\nPreprocessing data...")

   

    # Find the original maximum sequence length

    true_max_length = max(len(seq) for seq in sequences) if sequences else 0

    print(f"Original maximum sequence length: {true_max_length}")



    # Cap the sequence length to avoid memory issues from outliers

    max_length = min(true_max_length, MAX_SEQ_LENGTH_CAP)

    if true_max_length > max_length:

        print(f"Capping sequence length to {max_length}. Sequences longer than this will be truncated.")

   

    # Use keras utility to pad and truncate sequences

    # This is more efficient and handles both padding and truncating

    X = tf.keras.preprocessing.sequence.pad_sequences(

        sequences, maxlen=max_length, padding='post', truncating='post', dtype='float32'

    )

   

    print(f"Padded sequences shape: {X.shape}")

   

    # Convert text labels to numerical format using LabelEncoder

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(labels)

   

    # One-hot encode the labels using keras.utils.to_categorical

    y_categorical = keras.utils.to_categorical(y_encoded)

   

    print(f"Number of classes: {len(label_encoder.classes_)}")

    print(f"Classes: {label_encoder.classes_}")

    print(f"Labels shape: {y_categorical.shape}")

   

    # Add these debug prints before train_test_split

    print(f"Shape of X: {X.shape}")

    print(f"Shape of y_categorical: {y_categorical.shape}")

    print(f"Samples per class after encoding:")

    for i in range(y_categorical.shape[1]):

        print(f"Class {i}: {np.sum(y_categorical[:, i])}")

   

    # Determine test size, ensuring it's at least the number of classes

    num_classes = y_categorical.shape[1]

    num_samples = X.shape[0]

    test_size_ratio = 0.2

   

    # Check if the default test size is too small for stratification

    if num_samples * test_size_ratio < num_classes:

        # Calculate the minimum test size ratio needed for stratification

        min_test_size = num_classes / num_samples

        print(

            f"\nWarning: The dataset is too small for a standard 80/20 split. "

            f"Have {num_samples} samples and {num_classes} classes.\n"

            f"Adjusting test_size from {test_size_ratio:.2f} to {min_test_size:.2f} to ensure at least one sample per class in the test set."

        )

        test_size_ratio = min_test_size



    # Use stratified split

    X_train, X_test, y_train, y_test = train_test_split(

        X, y_categorical, test_size=test_size_ratio, random_state=42, stratify=y_categorical

    )

   

    print(f"Training set shape: {X_train.shape}")

    print(f"Testing set shape: {X_test.shape}")

   

    return X_train, X_test, y_train, y_test, label_encoder, max_length



def create_lstm_model(input_shape, num_classes):

    """

    Create an optimized LSTM model with bidirectional layers.

    When mixed precision is active the final Dense layer keeps float32
    for softmax numerical stability.
    """

    print(f"\nCreating LSTM model...")

    print(f"Input shape: {input_shape}")

    print(f"Number of classes: {num_classes}")

   

    # Determine output dtype — keep softmax in float32 under mixed precision
    output_dtype = "float32" if _device_mgr.is_mixed_precision_active else None

    model = keras.Sequential([

        layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=input_shape),

        layers.Dropout(0.3),

        

        layers.Bidirectional(layers.LSTM(128, return_sequences=False)),

        layers.Dropout(0.4),

       

        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),

        layers.BatchNormalization(),

        layers.Dropout(0.5),

        

        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),

        layers.Dropout(0.4),

       

        layers.Dense(num_classes, activation='softmax', dtype=output_dtype),

    ])

   

    return model



def train_model():

    """

    Main function to train the sign language recognition model.

    Uses the centralized DeviceManager for GPU acceleration, mixed
    precision, XLA JIT compilation, and tf.data pipeline optimisation.
    """

    print("Sign Language Recognition Model Training")

    print("=" * 50)

   

    # Load the data (now returns groups for leak-free splitting)
    sequences, labels, groups, sign_labels = load_sign_language_data()

    # ── CRITICAL FIX: split FIRST, then augment only training set ──
    # This prevents augmented copies from leaking into the test set.
    X_train_raw, X_test_raw, y_train_raw, y_test_raw, label_encoder, max_length = \
        preprocess_data(sequences, labels)

    # Augment training set only
    # Convert padded arrays back to list for augmentation, then re-pad
    import gc
    
    train_list = [X_train_raw[i] for i in range(len(X_train_raw))]
    train_labels = [label_encoder.inverse_transform([np.argmax(y_train_raw[i])])[0]
                    for i in range(len(y_train_raw))]
                    
    # EXPLICIT MEMORY CLEANUP: Free raw arrays before expanding data
    del X_train_raw
    del y_train_raw
    gc.collect()

    # Pass n_variants=1 to double the dataset size instead of quadrupling
    # (Memory safety first: 3500 * 2 = 7000 samples)
    train_list, train_labels = augment_dataset(train_list, train_labels, n_variants=1)

    # Re-pad augmented training data
    X_train = tf.keras.preprocessing.sequence.pad_sequences(
        train_list, maxlen=max_length, padding='post',
        truncating='post', dtype='float32'
    )
    y_train_encoded = label_encoder.transform(train_labels)
    y_train = keras.utils.to_categorical(y_train_encoded, num_classes=len(sign_labels))
    
    # EXPLICIT MEMORY CLEANUP: Free python lists now that numpy array is built
    del train_list
    del train_labels
    gc.collect()
    
    X_test = X_test_raw
    y_test = y_test_raw

    print(f"\nAfter augmentation:")
    print(f"  Train: {X_train.shape}")
    print(f"  Test:  {X_test.shape} (no augmentation — clean evaluation)")

   

    # ── Build model inside strategy scope for multi-GPU support ──
    input_shape = (max_length, X_train.shape[2])  # (sequence_length, features)

    num_classes = len(sign_labels)

   
    with _strategy.scope():
        model = create_lstm_model(input_shape, num_classes)

        print("\nCompiling model...")

        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)

        model.compile(

            optimizer=optimizer,

            loss='categorical_crossentropy',

            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]

        )

   

    # Print model summary

    print("\nModel Summary:")

    model.summary()

   
    # ── Build tf.data pipelines with prefetching ─────────────
    batch_size = DEFAULT_BATCH_SIZE
    train_ds = _device_mgr.build_dataset(X_train, y_train, batch_size=batch_size, training=True)
    val_ds = _device_mgr.build_dataset(X_test, y_test, batch_size=batch_size, training=False)

    callbacks = [

        keras.callbacks.EarlyStopping(

            monitor='val_loss',

            patience=15,

            restore_best_weights=True,

            verbose=1

        ),

        keras.callbacks.ReduceLROnPlateau(

            monitor='val_loss',

            factor=0.5,

            patience=7,

            min_lr=0.000001,

            verbose=1

        ),

        keras.callbacks.ModelCheckpoint(

            'models/best_model.keras',

            monitor='val_accuracy',

            save_best_only=True,

            verbose=1

        )

    ]

   

    print("\nStarting training...")
    print(f"  Device  : {_device_mgr.device_tag}")
    print(f"  Batch   : {batch_size}")
    print(f"  Mixed FP: {_device_mgr.is_mixed_precision_active}")

    history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=150,

        callbacks=callbacks,

        verbose=1

    )

   

    # Evaluate the model

    print("\nEvaluating model...")

    test_results = model.evaluate(X_test, y_test, verbose=0)
    test_loss, test_accuracy, test_top3_acc = test_results

    print(f"Test Loss: {test_loss:.4f}")

    print(f"Test Accuracy: {test_accuracy:.4f}")

    print(f"Test Top-3 Accuracy: {test_top3_acc:.4f}")

   
    # GPU memory summary (if available)
    _device_mgr.print_memory_usage()

    # Save the trained model to the models folder with the filename sign_model.keras

    models_dir = Path("models")

    models_dir.mkdir(exist_ok=True)

   

    model_path = models_dir / "sign_model.keras"

    model.save(model_path)

    print(f"\nModel saved to: {model_path}")

   

    # Save the label encoder for later use

    import pickle

    encoder_path = models_dir / "label_encoder.pkl"

    with open(encoder_path, 'wb') as f:

        pickle.dump(label_encoder, f)

    print(f"Label encoder saved to: {encoder_path}")



    # Save max_length for use in prediction

    max_length_path = models_dir / "max_length.txt"

    with open(max_length_path, 'w') as f:

        f.write(str(max_length))

    print(f"Max sequence length saved to: {max_length_path}")

   

    # Plot training history

    plot_training_history(history)

   

    return model, label_encoder, history



def plot_training_history(history):

    """

    Plot the training history (loss and accuracy).

    """

    print("\nPlotting training history...")

   

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

   

    # Plot training & validation loss

    ax1.plot(history.history['loss'], label='Training Loss')

    ax1.plot(history.history['val_loss'], label='Validation Loss')

    ax1.set_title('Model Loss')

    ax1.set_xlabel('Epoch')

    ax1.set_ylabel('Loss')

    ax1.legend()

    ax1.grid(True)

   

    # Plot training & validation accuracy

    ax2.plot(history.history['accuracy'], label='Training Accuracy')

    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')

    ax2.set_title('Model Accuracy')

    ax2.set_xlabel('Epoch')

    ax2.set_ylabel('Accuracy')

    ax2.legend()

    ax2.grid(True)

   

    plt.tight_layout()

   

    # Save the plot

    models_dir = Path("models")

    plot_path = models_dir / "training_history.png"

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    print(f"Training history plot saved to: {plot_path}")

   

    plt.show()



def predict_sign(model, label_encoder, sequence):

    """

    Predict the sign for a given sequence.

    """

    # Ensure sequence is the right shape

    if len(sequence.shape) == 2:

        sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])

   

    # Make prediction

    prediction = model.predict(sequence, verbose=0)

    predicted_class_idx = np.argmax(prediction[0])

    predicted_sign = label_encoder.inverse_transform([predicted_class_idx])[0]

    confidence = prediction[0][predicted_class_idx]

   

    return predicted_sign, confidence



if __name__ == "__main__":

    # Set random seeds for reproducibility

    np.random.seed(42)

    tf.random.set_seed(42)

   

    # Train the model

    model, label_encoder, history = train_model()

   

    print("\nTraining completed successfully!")

    print("You can now use the trained model for sign language recognition.")