import json
import time
from pathlib import Path

import numpy as np
import serial
import matplotlib.pyplot as plt

SIMULATE_BROKEN_LED = True
BROKEN_LED_AMOUNT = 1

SIMULATE_PARTTIME_BLOCKAGE = True
BLOCKAGE_AMOUNT = 1

SIMULATE_PARTIAL_BLOCKAGE = True
PORT = "COM11"       # change this
BAUD = 115200

DATA_NPY = "/dataset/exported/data_176.npy"  # shape [H, W, 36]
TIME_HOURS = 100000
TIMESTEP = 1000
SAMPLES_PER_TIMESTEP = 20
SEED = 42

STD = 0.005
FLICKERING_PROB = 0.001

ALPHA = 0.03
THERMAL_COEFF = 0.003


def valid_mask_from_data(data):
    valid = np.ones_like(data[:, :, 0], dtype=bool)
    valid[data[:, :, 0] == -1] = False
    return valid


def generate_relative_decay(timesteps, leds_n, rng):
    timestep= 1000
    hours_per_year = 365 * 24
    thermal_coeff = rng.uniform(0.002,1)
    alpha = rng.uniform(0.01, 0.05)
    relative_decay = np.ones((len(timesteps), leds_n))
    dirt_Decay = np.ones(len(timesteps))
    cleaned_on_hour = np.zeros(len(timesteps), dtype=bool)
    temperature = np.zeros(len(timesteps))

    decay_ks = rng.uniform(alpha - 0.01, alpha + 0.01, leds_n)
    relative_decay = np.ones((len(timesteps), len(decay_ks)))
    print(relative_decay.shape)
    aftectedLEDs = []
    if True:
        timesteps_years = timesteps / hours_per_year

        relative_decay = np.exp(-np.outer(timesteps_years, decay_ks))

    if True:
        angle = 90
        decay730 = 0  # decay after 730 days
        if angle == 90:
            decay730 = 0.015  # between 1-2.5
        hourly_decay = decay730 / (730 * 24)

        dirt_Decay = 1 - np.arange(0, len(timesteps)) * hourly_decay * timestep
        cleaned_on_hour = np.zeros(len(dirt_Decay), dtype=bool)
        decay730 = 0.08
        hourly_decay = decay730 / (730 * 24)
        DecayArray = 1 - np.arange(0, len(timesteps)) * hourly_decay * timestep
        dirt_Decay = dirt_Decay * DecayArray

    if True:
        clean_hour = round(len(cleaned_on_hour) / 2)
        cleaned_on_hour[clean_hour] = True
        dustathourx = dirt_Decay[clean_hour]
        dirt_Decay[clean_hour::] += 1 - dustathourx
    if True:
        hours = np.arange(len(timesteps)) * timestep
        # Assuming measurements at room temprature 20 degrees and if it is cooler more output at warmer tempratures decrease fo light outpu
        # so using the difference between assumed measurement teprature 20 and expected tempratures.
        yearlyTemp = 10 * np.sin(2 * np.pi * hours / (365 * 24))
        dailyTemp = 3 * np.sin(2 * np.pi * hours / 24)
        noiseTemp = rng.normal(0, 1, size=len(timesteps))
        thermalDroop = 1 - thermal_coeff * np.asarray(yearlyTemp + dailyTemp + noiseTemp)
        relative_decay *= thermalDroop[:, None]
        print("generating thermal droop")
        # todo
    if True:

        for i in range(1):
            DecayArray = np.zeros(len(timesteps))
            start = rng.integers(0, len(timesteps))
            DecayArray[start:-1] = 1
            random_led = rng.integers(0, leds_n)
            if (len(aftectedLEDs) == 36):
                raise ValueError(
                    f"affected_leds ({len(aftectedLEDs)}) cannot be larger than "
                    f"total_leds (36)"
                )
            while (aftectedLEDs.__contains__(random_led)):
                random_led = rng.integers(0, leds_n)
            aftectedLEDs.append(random_led)
            relative_decay[:, random_led] *= 1 - DecayArray
        # todo:
    if True:
        for i in range(1):
            DecayArray = np.zeros(len(timesteps))
            start = rng.integers(0, len(timesteps))
            DecayArray[start: rng.integers(start, len(timesteps)):-1] = 1
            random_led = rng.integers(0, leds_n)
            if (len(aftectedLEDs) == leds_n):
                raise ValueError(
                    f"affected_leds ({len(aftectedLEDs)}) cannot be larger than "
                    f"total_leds (36)"
                )
            while (aftectedLEDs.__contains__(random_led)):
                random_led = rng.integers(0, leds_n)
            aftectedLEDs.append(random_led)

            relative_decay[:, random_led] *= 1 - DecayArray
        # todo:
    if True:
        DecayArray = np.zeros(len(timesteps))
        DecayArray[rng.integers(0, len(timesteps)):-1] = rng.random()
        random_led = rng.integers(0, leds_n)
        if (len(aftectedLEDs) == leds_n):
            raise ValueError(
                f"affected_leds ({len(aftectedLEDs)}) cannot be larger than "
                f"total_leds (36)"
            )
        while (aftectedLEDs.__contains__(random_led)):
            random_led = rng.integers(0, leds_n)
        aftectedLEDs.append(random_led)

        relative_decay[:, random_led] *= 1 - DecayArray
        # todo:

    relative_decay = relative_decay * dirt_Decay[:, None]
    relative_decay[relative_decay < 0] = 0
    # Add noise to the data
    relative_decay += rng.normal(0, 0.005, size=relative_decay.shape)
    print(f"Generated {leds_n} decay constants and their decay scalers")
    return relative_decay, yearlyTemp + dailyTemp, cleaned_on_hour


def sample_positions(data, valid_mask, n, rng):
    H, W, leds_n = data.shape

    flat_data = data.reshape(H * W, leds_n)
    valid_flat_idxs = np.flatnonzero(valid_mask.reshape(-1))

    sample_flat_idxs = rng.choice(valid_flat_idxs, size=n)

    ys = sample_flat_idxs // W
    xs = sample_flat_idxs % W

    X = flat_data[sample_flat_idxs]
    y = np.stack([xs, ys], axis=1).astype(np.float32) * 10.0

    return X.astype(np.float32), y


def send_one_sample(ser, age, temperature, cleaned, x):
    cleaned_int = 1 if cleaned else 0

    values = [age, temperature, cleaned_int] + [float(v) for v in x]
    line = ",".join(f"{v:.8f}" for v in values) + "\n"

    ser.write(line.encode("utf-8"))
    ser.flush()


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
            # handles your older print format:
            # Prediction: x=2105.589355, y=986.995056
            clean = line.replace("Prediction:", "").replace("x=", "").replace("y=", "")
            parts = clean.replace(",", " ").split()
            return np.array([float(parts[0]), float(parts[1])], dtype=np.float32)

        if line.startswith("ERROR"):
            raise RuntimeError(line)


def main():
    rng = np.random.default_rng(SEED)

    X_base = np.load("X_test.npy").astype(np.float32)
    y_base = np.load("y_test.npy").astype(np.float32)

    leds_n = X_base.shape[1]
    assert leds_n == 36, f"Expected 36 features, got {leds_n}"

    timesteps = np.arange(0, TIME_HOURS, TIMESTEP)

    relative_decay, temperatures, cleaned_flags = generate_relative_decay(
        timesteps, leds_n, rng
    )

    errors = []
    all_results = []

    with serial.Serial(PORT, BAUD, timeout=2) as ser:
        time.sleep(2)

        for ti, t in enumerate(timesteps):
            sample_count = min(SAMPLES_PER_TIMESTEP, len(X_base))

            idxs = rng.choice(
                len(X_base),
                size=sample_count,
                replace=False
            )

            X_clean = X_base[idxs]
            y_true = y_base[idxs]

            decay = relative_decay[ti]
            X_degraded = X_clean * decay[None, :]

            flicker = rng.choice(
                [0, 1],
                size=X_degraded.shape,
                p=[FLICKERING_PROB, 1 - FLICKERING_PROB],
            )
            X_degraded *= flicker

            timestep_errors = []

            for i in range(sample_count):
                wait_for_ready(ser)

                send_one_sample(
                    ser,
                    age=float(t),
                    temperature=float(temperatures[ti]),
                    cleaned=bool(cleaned_flags[ti]),
                    x=X_degraded[i],
                )

                pred = wait_for_prediction(ser)
                err = np.linalg.norm(pred - y_true[i])

                timestep_errors.append(float(err))

                print(
                    f"t={t} sample={i} "
                    f"pred={pred} target={y_true[i]} err={err:.2f}"
                )

            avg_error = float(np.mean(timestep_errors))
            errors.append(avg_error)

            all_results.append({
                "timestep": float(t),
                "temperature": float(temperatures[ti]),
                "cleaned": bool(cleaned_flags[ti]),
                "average_error": avg_error,
            })

            print(f"Average error at {t} hours: {avg_error:.2f} mm")

    out_dir = Path("pico_degradation_results")
    out_dir.mkdir(exist_ok=True)
    avg_decay = relative_decay.mean(axis=1)
    min_decay = relative_decay.min(axis=1)
    max_decay = relative_decay.max(axis=1)
    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "timesteps": timesteps.tolist(),
            "errors": errors,
            "results": all_results,
        }, f, indent=4)

    # Plot positioning error
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, errors)
    plt.xlabel("Time in hours passed")
    plt.ylabel("Positioning Error (mm)")
    plt.title("Pico Positioning Error vs LED Degradation")
    plt.grid(True)
    plt.savefig(out_dir / "error_vs_degradation.png", dpi=150)
    plt.close()

    # Plot average/min/max degradation
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, avg_decay, label="Average degradation")
    plt.plot(timesteps, min_decay, label="Minimum degradation")
    plt.plot(timesteps, max_decay, label="Maximum degradation")
    plt.fill_between(timesteps, min_decay, max_decay, alpha=0.2)
    plt.xlabel("Time in hours passed")
    plt.ylabel("Decay scalar")
    plt.title("Average, Min, and Max LED Degradation Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "degradation_scalars.png", dpi=150)
    plt.close()
    print("Saved results to", out_dir)


if __name__ == "__main__":
    main()