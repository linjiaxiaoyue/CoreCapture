import os
os.environ['PYTHONHASHSEED'] = '42'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.keras.utils.set_random_seed(42)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

SEED = 42

X_TRAIN_PATH = "../Data/processed/X_train.npy"
Y_TRAIN_PATH = "../Data/processed/y_train.npy"

X_train = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)

print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")

def focal_loss(alpha, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        loss_vals = -alpha_t * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss_vals)
    return loss

def build_multi_branch_model(emb_dim=50, time_dim=2):
    input_emb = Input(shape=(emb_dim,), name='Embedding_Input')
    x1 = Dense(32, activation='relu')(input_emb)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.2)(x1)
    x1 = Dense(16, activation='relu')(x1)

    input_time = Input(shape=(time_dim,), name='Time_Input')
    x2 = Dense(8, activation='relu')(input_time)
    x2 = Dense(16, activation='relu')(x2)

    combined = Concatenate()([x1, x2])
    z = Dense(32, activation='relu')(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.2)(z)
    z = Dense(16, activation='relu')(z)

    output = Dense(1, activation='sigmoid', name='Output',bias_initializer=tf.keras.initializers.Constant(-6.9))(z)

    model = Model(inputs=[input_emb, input_time], outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss=focal_loss(alpha=0.75, gamma=2),
        metrics=[
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='prc', curve='PR')
        ]
    )
    return model

tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(SEED)

classes_final = np.unique(y_train)
cw_final = compute_class_weight(class_weight='balanced', classes=classes_final, y=y_train)
class_weights_final = {int(c): float(w) for c, w in zip(classes_final, cw_final)}
print(f"Class weights: {class_weights_final}")

X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train,
    test_size=0.15,
    random_state=SEED,
    stratify=y_train
)

X_train_emb = X_train_split[:, :50]
X_train_time = X_train_split[:, 50:]
X_val_emb = X_val[:, :50]
X_val_time = X_val[:, 50:]

os.makedirs('model_checkpoints', exist_ok=True)
checkpoint_path = "model_checkpoints/best_model.h5"

checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_prc',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

model = build_multi_branch_model(emb_dim=50, time_dim=2)

print("\nStarting training...")
final = model.fit(
    [X_train_emb, X_train_time], y_train_split,
    validation_data=([X_val_emb, X_val_time], y_val),
    epochs=100,
    batch_size=64,
    class_weight=class_weights_final,
    callbacks=[checkpoint_callback],
    verbose=1
)
print("\nTraining completed!")