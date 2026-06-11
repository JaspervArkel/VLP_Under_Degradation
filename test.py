import time
import numpy as np
import serial
import tensorflow as tf

PORT = "COM11"
BAUD = 115200
idx = 0

X = np.load("X_test.npy").astype(np.float32)
y = np.load("y_test.npy").astype(np.float32)

x_mean = np.load("x_mean.npy").astype(np.float32)
x_std = np.load("x_std.npy").astype(np.float32)

sample = X[idx:idx + 1]
target = y[idx]

# ---------- PC TFLite prediction ----------
sample_norm = (sample - x_mean) / x_std

interpreter = tf.lite.Interpreter(model_path="tiny_pico_model_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero = input_details["quantization"]
output_scale, output_zero = output_details["quantization"]

sample_q = sample_norm / input_scale + input_zero
sample_q = np.clip(np.round(sample_q), -128, 127).astype(np.int8)

interpreter.set_tensor(input_details["index"], sample_q)
interpreter.invoke()

out_q = interpreter.get_tensor(output_details["index"])
pc_pred = (out_q.astype(np.float32) - output_zero) * output_scale

print("Target:", target)
print("PC TFLite prediction:", pc_pred[0])
print("PC TFLite error:", np.linalg.norm(pc_pred[0] - target))
print("First raw values:", sample[0, :4])
print("First normalized values:", sample_norm[0, :4])
print("First quantized values:", sample_q[0, :4])

# ---------- Send exact same raw sample to Pico ----------
age = 0.0
temperature = 0.0
cleaned = 0

values = [age, temperature, cleaned] + [float(v) for v in sample[0]]
msg = ",".join(f"{v:.8f}" for v in values) + "\n"

def wait_for_ready(ser):
    while True:
        line = ser.readline().decode(errors="replace").strip()
        if line:
            print("Pico:", line)
        if line == "READY" or "Send 36" in line:
            return

def wait_for_prediction(ser):
    while True:
        line = ser.readline().decode(errors="replace").strip()
        if line:
            print("Pico:", line)

        if line.startswith("PRED,"):
            _, px, py = line.split(",")
            return np.array([float(px), float(py)], dtype=np.float32)

        if line.startswith("Prediction:"):
            clean = line.replace("Prediction:", "").replace("x=", "").replace("y=", "")
            parts = clean.replace(",", " ").split()
            return np.array([float(parts[0]), float(parts[1])], dtype=np.float32)

        if line.startswith("ERROR"):
            raise RuntimeError(line)

with serial.Serial(PORT, BAUD, timeout=2) as ser:
    time.sleep(2)

    print("Waiting for Pico...")
    wait_for_ready(ser)

    print("Sending exact same raw sample to Pico...")
    ser.write(msg.encode("utf-8"))
    ser.flush()

    pico_pred = wait_for_prediction(ser)

print("Pico prediction:", pico_pred)
print("Pico error:", np.linalg.norm(pico_pred - target))
print("Difference PC TFLite vs Pico:", np.linalg.norm(pc_pred[0] - pico_pred))
print("PY RAW4:", sample[0, :4])
print("PY NORM4:", sample_norm[0, :4])
print("PY Q4:", sample_q[0, :4])
print("PY input scale/zero:", input_scale, input_zero)
print("PY output scale/zero:", output_scale, output_zero)