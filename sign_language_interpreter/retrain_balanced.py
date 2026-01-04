import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def load_data():
    mp_data_path = Path("MP_DATA_QUALITY")
    sequences, labels = [], []
    
    for class_dir in mp_data_path.iterdir():
        if class_dir.is_dir():
            for npy_file in class_dir.glob("*.npy"):
                sequences.append(np.load(npy_file).astype(np.float32))
                labels.append(class_dir.name)
    
    return sequences, labels

def augment_data(sequences, labels):
    aug_seq, aug_labels = list(sequences), list(labels)
    for seq, label in zip(sequences, labels):
        for _ in range(2):
            noise = np.random.normal(0, 0.005, seq.shape).astype(np.float32)
            aug_seq.append(seq + noise)
            aug_labels.append(label)
    return aug_seq, aug_labels

def preprocess_data(sequences, labels):
    max_length = min(max(len(s) for s in sequences), 60)
    X = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_length, padding='post', truncating='post', dtype='float32'
    )
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    y_categorical = keras.utils.to_categorical(y_encoded)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
    )
    
    return X_train, X_test, y_train, y_test, label_encoder, max_length, y_encoded

def create_model(input_shape, num_classes):
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

def train():
    print("="*70)
    print("TRAINING WITH CLASS WEIGHTS (BALANCED)")
    print("="*70)
    
    sequences, labels = load_data()
    print(f"Loaded: {len(sequences)} samples, {len(set(labels))} classes")
    
    sequences, labels = augment_data(sequences, labels)
    print(f"After augmentation: {len(sequences)} samples")
    
    X_train, X_test, y_train, y_test, label_encoder, max_length, y_encoded_all = preprocess_data(sequences, labels)
    
    # Compute class weights to balance classes
    class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded_all), y=y_encoded_all)
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"\n✓ Using class weights to balance training")
    print(f"  Weight range: {min(class_weights):.2f} - {max(class_weights):.2f}")
    
    model = create_model((max_length, X_train.shape[2]), len(label_encoder.classes_))
    
    model.compile(
        optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint('models/best_model.keras', monitor='val_accuracy', save_best_only=True)
    ]
    
    print("\nTraining with balanced class weights...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=64,
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['loss'], label='Train')
    ax1.plot(history.history['val_loss'], label='Val')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['accuracy'], label='Train')
    ax2.plot(history.history['val_accuracy'], label='Val')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(models_dir / "training_history.png", dpi=300)
    print(f"\n✅ Balanced model saved!")

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    train()
