import numpy as np

X = np.load("X_test.npy")
y = np.load("y_test.npy")

idx = 0

print("Expected target:", y[idx])
print("float sample[36] = {")
for v in X[idx]:
    print(f"    {v:.8f}f,")
print("};")