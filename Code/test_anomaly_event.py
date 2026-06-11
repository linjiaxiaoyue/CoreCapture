import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

SEED = 42
os.environ['PYTHONHASHSEED'] = '42'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import random
random.seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

def focal_loss(alpha, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        loss_vals = -alpha_t * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss_vals)
    return loss

MODEL_PATH = "model_checkpoints/best_model.h5"
TEST_X_PATH = "../Data/processed/X_test.npy"
TEST_Y_PATH = "../Data/processed/y_test.npy"
WINDOW_NAMES_PATH = "../Data/processed/window_names.npy"

print("Loading model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "loss": focal_loss(alpha=0.75, gamma=2),
        "focal_loss": focal_loss(alpha=0.75, gamma=2)
    }
)
print("Model loaded successfully!\n")

print("Loading test data...")
X_test = np.load(TEST_X_PATH)
y_test = np.load(TEST_Y_PATH)
window_names = np.load(WINDOW_NAMES_PATH, allow_pickle=True)

print(f"Test set shape: X={X_test.shape}, y={y_test.shape}, window_names={window_names.shape}")

X_test_emb = X_test[:, :50]
X_test_time = X_test[:, 50:]

print("\nStarting prediction...")
y_pred_proba = model.predict([X_test_emb, X_test_time], batch_size=64, verbose=1)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()
y_true = y_test.flatten()

print("\n" + "="*40)
print("Test Set Evaluation Results")
print("="*40)
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("="*40)

print("\n" + "="*50)
print("Unique Window Names Classified as ANOMALY by Model")
print("="*50)

anomaly_indices = np.where(y_pred == 1)[0]
anomaly_window_names = window_names[anomaly_indices]

unique_anomaly_windows = []
seen = set()
for name in anomaly_window_names:
    if name not in seen:
        seen.add(name)
        unique_anomaly_windows.append(name)

print(f"Total samples classified as anomaly: {len(anomaly_indices)}")
print(f"Number of unique anomaly window names: {len(unique_anomaly_windows)}")
print("\nAnomaly window names list:\n")
for i, name in enumerate(unique_anomaly_windows, 1):
    print(f"{i:2d}. {name}")

print("\n" + "="*50)
print("Testing completed!")