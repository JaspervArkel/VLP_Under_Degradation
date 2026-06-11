import numpy as np
import tensorflow as tf


def anti_age(X, age, alpha=0.03):
    return X * (1.0 / (1.0 - alpha / (365.0 * 24.0))) ** age


def anti_dust(X, age, last_cleaned=0):
    dust_collection_time = age - last_cleaned

    expected_reduction_sensor = (1.0 - 0.015 / (730.0 * 24.0)) ** dust_collection_time
    expected_reduction_lights = (1.0 - 0.08 / (730.0 * 24.0)) ** dust_collection_time

    return X * (1.0 / expected_reduction_lights) * (1.0 / expected_reduction_sensor)


def anti_thermal_droop(X, temperature, coefficient=0.003):
    change = 1.0 / (1.0 - coefficient * temperature)
    return X * change


def anti_degrade(X, temperature=0.0, age=0.0, alpha=0.03, cleaned=False):
    # For training, keep this simple.
    # If cleaned=True, you can reset age externally in your dataset logic.
    X = anti_age(X, age, alpha)
    X = anti_dust(X, age)
    X = anti_thermal_droop(X, temperature)
    return X


# ----------------------------------------------------
# Load your data here.
# You need:
# X_train shape: [num_samples, 36]
# y_train shape: [num_samples, 2]
# ----------------------------------------------------

X_train = np.load("X_train.npy").astype(np.float32)
y_train = np.load("y_train.npy").astype(np.float32)

# Optional: apply degradation correction before training.
# Use real temperature/age arrays if you have them.
temperature = 0.0
age = 0.0
X_train = anti_degrade(X_train, temperature=temperature, age=age)

# Save normalization info.
# This is easier for Pico than L2 norm inside the neural net.
x_mean = X_train.mean(axis=0).astype(np.float32)
x_std = (X_train.std(axis=0) + 1e-8).astype(np.float32)

np.save("x_mean.npy", x_mean)
np.save("x_std.npy", x_std)

X_train_norm = (X_train - x_mean) / x_std

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(36,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(2),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"],
)

model.fit(
    X_train_norm,
    y_train,
    batch_size=64,
    epochs=100,
    validation_split=0.1,
    shuffle=True,
)

model.save("tiny_pico_model.keras")

# ----------------------------------------------------
# Convert to int8 TFLite
# ----------------------------------------------------

def representative_dataset():
    for i in range(min(500, len(X_train_norm))):
        sample = X_train_norm[i:i + 1].astype(np.float32)
        yield [sample]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("tiny_pico_model_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved tiny_pico_model_int8.tflite")
print("Saved x_mean.npy and x_std.npy")