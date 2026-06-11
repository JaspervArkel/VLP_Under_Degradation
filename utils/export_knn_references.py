import numpy as np

N_REF = 200
SEED = 42

X = np.load("X_train.npy").astype(np.float32)
y = np.load("y_train.npy").astype(np.float32)

rng = np.random.default_rng(SEED)
idx = rng.choice(len(X), size=min(N_REF, len(X)), replace=False)

X_ref = X[idx]
y_ref = y[idx]

with open("knn_reference_data.h", "w") as f:
    f.write("#ifndef KNN_REFERENCE_DATA_H_\n")
    f.write("#define KNN_REFERENCE_DATA_H_\n\n")

    f.write(f"#define KNN_REF_N {len(X_ref)}\n")
    f.write("#define KNN_K 5\n\n")

    f.write("const float KNN_TRAIN_X[KNN_REF_N][36] = {\n")
    for row in X_ref:
        values = ", ".join(f"{float(v):.8f}f" for v in row)
        f.write(f"    {{{values}}},\n")
    f.write("};\n\n")

    f.write("const float KNN_TRAIN_Y[KNN_REF_N][2] = {\n")
    for row in y_ref:
        values = ", ".join(f"{float(v):.8f}f" for v in row)
        f.write(f"    {{{values}}},\n")
    f.write("};\n\n")

    f.write("#endif\n")

print("Wrote knn_reference_data.h")