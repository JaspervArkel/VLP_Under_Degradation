import numpy as np
import tensorflow as tf

X_test = np.load("X_train.npy").astype(np.float32)
y_test = np.load("y_train.npy").astype(np.float32)

x_mean = np.load("x_mean.npy").astype(np.float32)
x_std = np.load("x_std.npy").astype(np.float32)

X_test = (X_test - x_mean) / x_std

interpreter = tf.lite.Interpreter(model_path="tiny_pico_model_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero_point = input_details["quantization"]
output_scale, output_zero_point = output_details["quantization"]

errors = []

for i in range(min(100, len(X_test))):
    x = X_test[i:i + 1]

    x_q = x / input_scale + input_zero_point
    x_q = np.clip(np.round(x_q), -128, 127).astype(np.int8)

    interpreter.set_tensor(input_details["index"], x_q)
    interpreter.invoke()

    y_q = interpreter.get_tensor(output_details["index"])
    y_pred = (y_q.astype(np.float32) - output_zero_point) * output_scale

    err = np.linalg.norm(y_pred[0] - y_test[i])
    errors.append(err)

print("Mean error:", np.mean(errors))
print("First prediction:", y_pred[0])
print("First target:", y_test[0])