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

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU detected: {len(gpus)} GPU(s) available")
        print(f"  {gpus}")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")
else:
    print("⚠ No GPU detected - using CPU (training will be slower)")

# Add this constant at the top of the file after imports

MIN_SAMPLES_PER_CLASS = 10



def load_sign_language_data():

    """

    Load sign language data from MP_DATA folder.

    Returns sequences (X) and labels (y) for training.

    Filters out classes with too few samples.

    """

    print("Loading sign language data...")

   

    # Define path to MP_DATA folder

    mp_data_path = Path("MP_DATA")

   

    if not mp_data_path.exists():

        raise FileNotFoundError("MP_DATA folder not found. Please run process_videos.py first.")

   

    sequences = []

    labels = []

   

    # Get all sign label directories

    sign_labels = [d.name for d in mp_data_path.iterdir() if d.is_dir()]

    print(f"Found sign labels: {sign_labels}")

   

    # Load data from each sign label folder

    for sign_label in sign_labels:

        sign_folder = mp_data_path / sign_label

        npy_files = list(sign_folder.glob("*.npy"))

       

        print(f"Loading {len(npy_files)} sequences for '{sign_label}'")

       

        for npy_file in npy_files:

            # Load the sequence data, ensuring it's float32 to save memory

            sequence = np.load(npy_file).astype(np.float32)

            sequences.append(sequence)

            labels.append(sign_label)

   

    print(f"\nTotal sequences loaded: {len(sequences)}")

    print(f"Sequence shape: {sequences[0].shape}")

   

    # Count samples per class

    label_counts = Counter(labels)

    print("\nSamples per class:")

    for label, count in label_counts.items():

        print(f"{label}: {count}")

   

    # Filter out classes with too few samples

    filtered_sequences = []

    filtered_labels = []

    removed_classes = set()

    kept_classes = set()

   

    for sequence, label in zip(sequences, labels):

        if label_counts[label] >= MIN_SAMPLES_PER_CLASS:

            filtered_sequences.append(sequence)

            filtered_labels.append(label)

            kept_classes.add(label)

        else:

            removed_classes.add(label)

   

    print(f"\nFiltering classes with less than {MIN_SAMPLES_PER_CLASS} samples:")

    print(f"Classes kept ({len(kept_classes)}): {sorted(kept_classes)}")

    print(f"Classes removed ({len(removed_classes)}): {sorted(removed_classes)}")

    print(f"Sequences after filtering: {len(filtered_sequences)}")

   

    return filtered_sequences, filtered_labels, sorted(kept_classes)



def augment_data(sequences, labels, noise_level=0.005, num_augmentations=2):

    """

    Augment the data by adding small random noise to each sequence.

    """

    print("\nAugmenting data...")

    augmented_sequences = list(sequences)

    augmented_labels = list(labels)

   

    for sequence, label in zip(sequences, labels):

        for _ in range(num_augmentations):

            noise = np.random.normal(0, noise_level, sequence.shape).astype(np.float32)

            augmented_sequences.append(sequence + noise)

            augmented_labels.append(label)

       

    print(f"Augmented from {len(sequences)} to {len(augmented_sequences)} sequences")

    return augmented_sequences, augmented_labels



MAX_SEQ_LENGTH_CAP = 60



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

    """

    print(f"\nCreating LSTM model...")

    print(f"Input shape: {input_shape}")

    print(f"Number of classes: {num_classes}")

   

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

       

        layers.Dense(num_classes, activation='softmax')

    ])

   

    return model



def train_model():

    """

    Main function to train the sign language recognition model.

    """

    print("Sign Language Recognition Model Training")

    print("=" * 50)

   

    # Load the data

    sequences, labels, sign_labels = load_sign_language_data()

   

    # Augment only if needed

    if len(sequences) < 1000:

        sequences, labels = augment_data(sequences, labels, num_augmentations=2)

    print("Samples per class after augmentation:")

    for label, count in Counter(labels).most_common():

        print(f"{label}: {count}")

   

    # Preprocess the data

    X_train, X_test, y_train, y_test, label_encoder, max_length = preprocess_data(sequences, labels)

   

    # Create the model

    input_shape = (max_length, X_train.shape[2])  # (sequence_length, features)

    num_classes = len(sign_labels)

   

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

    batch_size = 64

    history = model.fit(

        X_train, y_train,

        validation_data=(X_test, y_test),

        epochs=150,

        batch_size=batch_size,

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