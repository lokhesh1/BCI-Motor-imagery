# mainv2.py
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from dataloadv2 import load_and_preprocess_data

print("EEG Motor Signal Detector Training")

# ----------------------
# Hyperparameters
# ----------------------
HYPERPARAMS = {
    "config_default": {
        "learning_rate": 0.001,
        "dropout_rate": 0.3,
        "l2_regularization": 0.002,
        "batch_size": 32,
        "epochs": 200
    },
    "config_balanced_reg": {
        "learning_rate": 0.0005,
        "dropout_rate": 0.4,
        "l2_regularization": 0.003,
        "batch_size": 32,
        "epochs": 200
    },
    "config_higher_bs": {
        "learning_rate": 0.0005,
        "dropout_rate": 0.5,
        "l2_regularization": 0.002,
        "batch_size": 64,
        "epochs": 200
    },
}

CONFIG_NAME = "config_balanced_reg"
cfg = HYPERPARAMS[CONFIG_NAME]
print(f"Using config: {CONFIG_NAME}")

# ----------------------
# Load Data
# ----------------------
X, y, num_classes, input_shape = load_and_preprocess_data("data/train")
if X.size == 0 or num_classes < 2:
    print("Insufficient data for training.")
    exit()

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------
# Model Architecture
# ----------------------
def CNN_LSTM(input_shape, num_classes, dropout_rate, l2_reg):
    inp = Input(shape=input_shape)
    x = Conv1D(32, 3, padding='same', activation='elu', kernel_regularizer=l2(l2_reg))(inp)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(64, 3, padding='same', activation='elu', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(dropout_rate)(x)

    x = LSTM(128, return_sequences=True, dropout=dropout_rate)(x)
    x = LSTM(64, dropout=dropout_rate)(x)

    x = Dense(64, activation='elu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inp, out)

model = CNN_LSTM(input_shape, num_classes, cfg['dropout_rate'], cfg['l2_regularization'])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=cfg['learning_rate']),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------
# Training
# ----------------------
early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, mode='max')
lr_sched = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=cfg['epochs'],
    batch_size=cfg['batch_size'],
    callbacks=[early_stop, lr_sched],
    verbose=1
)

# Save history
with open(f"training_history_{CONFIG_NAME}.json", 'w') as f:
    json.dump({k: list(map(float, v)) for k, v in history.history.items()}, f)

# ----------------------
# Evaluation & Save Model
# ----------------------
loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss: {loss:.4f}, Accuracy: {acc:.4f}")

model.save(f"cnn_lstm_eeg_motor_{CONFIG_NAME}.h5")
print("Training complete.")
