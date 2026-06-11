from pathlib import Path

model_path = Path("tiny_pico_model_int8.tflite")
out_path = Path("model_data.cc")

if not model_path.exists():
    raise FileNotFoundError(f"Could not find {model_path.resolve()}")

data = model_path.read_bytes()

with out_path.open("w", encoding="utf-8") as f:
    f.write('#include "model_data.h"\n\n')
    f.write("alignas(16) const unsigned char g_model[] = {\n")

    for i in range(0, len(data), 12):
        chunk = data[i:i + 12]
        hex_values = ", ".join(f"0x{b:02x}" for b in chunk)
        f.write(f"    {hex_values},\n")

    f.write("};\n\n")
    f.write(f"const int g_model_len = {len(data)};\n")

print(f"Wrote {out_path} with {len(data)} bytes")