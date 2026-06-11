import numpy as np

x_mean = np.load("x_mean.npy").astype(np.float32)
x_std = np.load("x_std.npy").astype(np.float32)

with open("normalization_data.h", "w") as f:
    f.write("#ifndef NORMALIZATION_DATA_H_\n")
    f.write("#define NORMALIZATION_DATA_H_\n\n")

    f.write("const float X_MEAN[36] = {\n")
    f.write(",\n".join([f"    {v:.8f}f" for v in x_mean]))
    f.write("\n};\n\n")

    f.write("const float X_STD[36] = {\n")
    f.write(",\n".join([f"    {v:.8f}f" for v in x_std]))
    f.write("\n};\n\n")

    f.write("#endif\n")