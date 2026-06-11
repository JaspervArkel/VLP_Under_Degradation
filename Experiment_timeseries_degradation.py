import argparse
import json
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from numpy.f2py.auxfuncs import throw_error
from tqdm import tqdm

from dataset.led_age_series_util import calculate_decay_constant
from experiment import task_registry
from models import register_models, MLPAntiDegradation
from models.rf_anti_degradation import RFAntiDegradation

R90_MIN = 10000
R90_MAX = 50000


def generate_test_set(data: npt.NDArray, valid_mask: npt.NDArray, test_set_fraction: float, rng: np.random.Generator) -> \
tuple[npt.NDArray, npt.NDArray]:
    H, W, leds_n = data.shape

    # Flatten data
    flat_data = data.reshape(W * H, leds_n)

    # Generate LED ids
    led_ids = np.arange(leds_n)[None, :]  # Shape (1, 1, leds_n)
    # Generate random sample indices at each time step
    valid_flat_idxs = np.flatnonzero(valid_mask)

    sample_flat_idxs = rng.choice(valid_flat_idxs, int(H * W * test_set_fraction))

    # Calculate the x and y coordinates of the samples
    ys = sample_flat_idxs // W
    xs = sample_flat_idxs % W
    sample_locs = np.stack((xs, ys), axis=-1)

    # Fetch the samples for each LED at each time step
    # Add a new axis to the sample_flat_idxs to match the shape of led_ids
    sample_flat_idxs = sample_flat_idxs[:, None]

    # Get samples
    # Get the samples for each LED at each timestep with the same sample index. Shape (H*W*fraction, leds_n)
    samples = flat_data[sample_flat_idxs, led_ids]

    return samples, sample_locs


def generate_degraded_samples(data: npt.NDArray, valid_mask: npt.NDArray, relative_decay: npt.NDArray,
                          samples_per_timestep: int, rng: np.random.Generator) -> tuple[npt.NDArray, npt.NDArray]:
    timesteps_n, leds_n = relative_decay.shape
    H, W, leds_n = data.shape

    # Flatten data
    flat_data = data.reshape(W * H, leds_n)

    # Generate LED ids
    led_ids = np.arange(leds_n)[None, None, :]  # Shape (1, 1, leds_n)
    # Generate random sample indices at each time step
    valid_flat_idxs = np.flatnonzero(valid_mask)

    sample_flat_idxs = rng.choice(
        valid_flat_idxs, size=(timesteps_n, samples_per_timestep))

    # Calculate the x and y coordinates of the samples
    ys = sample_flat_idxs // W
    xs = sample_flat_idxs % W
    sample_locs = np.stack((xs, ys), axis=-1)

    # Fetch the samples for each LED at each time step
    # Add a new axis to the sample_flat_idxs to match the shape of led_ids
    sample_flat_idxs = sample_flat_idxs[:, :, None]
    sample_flat_idxs = np.broadcast_to(
        sample_flat_idxs, (timesteps_n, samples_per_timestep, leds_n))
    led_ids = np.broadcast_to(
        led_ids, (timesteps_n, samples_per_timestep, leds_n))

    # Age the samples
    # Get the samples for each LED at each timestep with the same sample index. Shape (timesteps, samples_per_timestep, leds_n)
    samples = flat_data[sample_flat_idxs, led_ids]
    # Apply the relative decay to the samples
    aged_samples = samples * relative_decay[:, None, :]

    # Add flickering
    flickering = rng.choice([0, 1], size=aged_samples.shape, p=[args.flickering_prob, 1 - args.flickering_prob])
    aged_samples = aged_samples * flickering  # Apply flickering to the samples
    print(f"Applied flickering to {(flickering == 0).sum()} samples")

    return aged_samples, sample_locs





def gen_relative_decay(timesteps ,rng, leds_n, timestep, alpha , simulate_dirt_cleaning,
                       simulateLEDDirt ,generateThermalDroop, simulateBrokenLed , simulateparttimeblockages ,
                       simulatePartitionalBlockage , BlockageAmount , brokenAmount, simulateAging, specifiedBrokenLedTimes = None):
    hours_per_year = 365 * 24
    thermal_coeff = rng.uniform(0.002, 0.01)
    relative_decay = np.ones((len(timesteps), leds_n))
    dirt_Decay = np.ones(len(timesteps))
    cleaned_on_hour = np.zeros(len(timesteps), dtype=bool)
    temperature = np.zeros(len(timesteps))

    decay_ks = rng.uniform(alpha - 0.01, alpha + 0.01, leds_n)
    relative_decay = np.ones((len(timesteps), len(decay_ks)))
    print(relative_decay.shape)
    aftectedLEDs = []
    if simulateAging:
        timesteps_years = timesteps / hours_per_year

        relative_decay = np.exp(-np.outer(timesteps_years, decay_ks))

    if simulateLEDDirt:
        angle = 90
        decay730 = 0 #decay after 730 days
        if angle == 90:
            decay730 = 0.015 #between 1-2.5
        hourly_decay = decay730/(730*24)

        dirt_Decay = 1- np.arange(0, len( timesteps) ) *hourly_decay  * timestep
        cleaned_on_hour = np.zeros(len(dirt_Decay), dtype=bool)
        decay730 = 0.08
        hourly_decay = decay730 / (730 * 24)
        DecayArray = 1 - np.arange(0, len(timesteps)) *hourly_decay  * timestep
        dirt_Decay = dirt_Decay*DecayArray

    if simulate_dirt_cleaning:

        clean_hour = round(len(cleaned_on_hour)/2)
        cleaned_on_hour[clean_hour] = True
        dustathourx = dirt_Decay[clean_hour]
        dirt_Decay[clean_hour::] += 1-dustathourx
    if generateThermalDroop:
        hours = np.arange(len(timesteps)) * timestep
        #Assuming measurements at room temprature 20 degrees and if it is cooler more output at warmer tempratures decrease fo light outpu
        # so using the difference between assumed measurement teprature 20 and expected tempratures.
        yearlyTemp = 10*np.sin(2 *np.pi *hours / (365*24))
        dailyTemp = 3 *np.sin(2* np.pi * hours / 24)
        noiseTemp = rng.normal(0,1,size = len(timesteps))
        thermalDroop = 1- thermal_coeff * np.asarray(yearlyTemp + dailyTemp + noiseTemp)
        relative_decay *= thermalDroop[:, None]
        print("generating thermal droop")
        #todo
    if simulateBrokenLed:


        for i in range(brokenAmount):
            DecayArray = np.zeros(len(timesteps))
            start =rng.integers(0, len(timesteps))
            DecayArray[start:-1] =1
            random_led = rng.integers(0, leds_n)
            if (len(aftectedLEDs) == 36):
                raise ValueError(
                    f"affected_leds ({len(aftectedLEDs)}) cannot be larger than "
                    f"total_leds (36)"
                )
            while( aftectedLEDs.__contains__(random_led) ):
                random_led = rng.integers(0, leds_n)
            aftectedLEDs.append(random_led)
            relative_decay[:, random_led] *= 1- DecayArray
        #todo:
    if simulateparttimeblockages:
        for i in range(BlockageAmount):
            DecayArray = np.zeros(len(timesteps))
            start= rng.integers(0, len(timesteps))
            DecayArray[start : rng.integers(start, len(timesteps)) :-1] =1
            random_led = rng.integers(0, leds_n)
            if(len(aftectedLEDs)==leds_n):
                raise ValueError(
                    f"affected_leds ({len(aftectedLEDs)}) cannot be larger than "
                    f"total_leds (36)"
                )
            while( aftectedLEDs.__contains__(random_led) ):
                random_led = rng.integers(0, leds_n)
            aftectedLEDs.append(random_led)

            relative_decay[:, random_led] *= 1- DecayArray
        # todo:
    if simulatePartitionalBlockage:
        DecayArray = np.zeros(len(timesteps))
        DecayArray[rng.integers(0,  len(timesteps)) :-1] = rng.random()
        random_led = rng.integers(0, leds_n)
        if(len(aftectedLEDs)==leds_n):
            raise ValueError(
                f"affected_leds ({len(aftectedLEDs)}) cannot be larger than "
                f"total_leds (36)"
            )
        while( aftectedLEDs.__contains__(random_led) ):
            random_led = rng.integers(0, leds_n)
        aftectedLEDs.append(random_led)

        relative_decay[:, random_led] *= 1- DecayArray
        # todo:



    relative_decay = relative_decay * dirt_Decay[:, None]
    relative_decay[relative_decay < 0] = 0
    # Add noise to the data
    relative_decay += rng.normal(0, args.std, size=relative_decay.shape)
    print(f"Generated {leds_n} decay constants and their decay scalers")
    return relative_decay, decay_ks , yearlyTemp + dailyTemp ,cleaned_on_hour


def main(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    task = args.task
    tasks = task_registry.get_task_names()
    if task not in tasks:
        raise ValueError(f"Task {task} not found. Available tasks: {tasks}")

    print("Loading model...")
    model_cls = task_registry.get_model_class(args.task)
    model_args = task_registry.get_model_args(args.task)
    model = model_cls(**model_args, device=args.device, seed=args.seed)

    model.load(args.load_model)
    print(f"Model loaded from {args.load_model}")

    data = np.load(args.src)
    H, W, leds_n = data.shape
    print(f"Data loaded from {args.src}")

    # Generate valid mask for data
    # Shape (H, W), 1 for valid, 0 for invalid same for all LEDs
    valid_mask = np.ones_like(data[:, :, 0], dtype=float)
    valid_mask[data[:, :, 0] == -1] = 0
    print(f"Generated valid mask ({valid_mask.sum()} valid points)")

    # Generate time steps
    timesteps = np.arange(0, args.time, args.timestep)
    print(f"Generated {len(timesteps)} timesteps")

    ledDirt = args.simulate_LED_Dirt
    thermaldroop =args.simulate_thermal_droop
    simulate_broken_led=args.simulate_broken_led
    simulate_dirt_cleaning=args.simulate_dirt_cleaning
    simulate_partime_blockages=args.simulate_parttime_blockage
    simulate_partitional_blockages=args.simulate_partitional_blockage
    alpha = rng.uniform(0.01, 0.05)
    relative_decay ,decay_ks, temperature , cleaned = gen_relative_decay(timesteps, rng, leds_n, args.timestep,alpha,
                                                  simulate_dirt_cleaning, ledDirt,
                                                    thermaldroop,simulate_broken_led ,
                                                  simulate_partime_blockages,simulate_partitional_blockages, args.blockage_amount,
                                                  args.broken_LED_amount,args.simulateAging
                                                  )


    aged_samples_X, aged_samples_y = generate_degraded_samples(
        data, valid_mask, relative_decay, args.samples_per_timestep, rng)
    aged_samples_X, aged_samples_y = torch.tensor(
        aged_samples_X, dtype=torch.float, device=args.device), torch.tensor(aged_samples_y, dtype=torch.float,
                                                                             device=args.device) * 10
    print(
        f"Generated {aged_samples_X.shape[0] * aged_samples_X.shape[1]} samples; {aged_samples_X.shape[1]} samples at {aged_samples_X.shape[0]} timesteps")

    test_X, test_y = generate_test_set(data, valid_mask, 0.2, rng)

    # Translate to PyTorch
    test_X, test_y = torch.tensor(test_X, dtype=torch.float, device=args.device), torch.tensor(
        test_y, dtype=torch.float, device=args.device) * 10
    relative_decay = torch.tensor(relative_decay, dtype=torch.float, device=args.device)

    errors = []
    bar = tqdm(enumerate(timesteps), total=len(timesteps))
    age = 0
    for i, t in bar:
        # Warm up the model with samples at this timestep
        aged_samples_X_t, aged_samples_y_t = aged_samples_X[i,
        :, :], aged_samples_y[i, :, :]

        if isinstance(model, MLPAntiDegradation) or isinstance(model, RFAntiDegradation):
            sample_predictions = model.predict(aged_samples_X_t, age = age  *args.timestep, temperature = temperature[age], cleaned = cleaned[age])
            print(temperature[age])
            print("anti degradation")

        else:
            sample_predictions = model.predict(aged_samples_X_t, )
        # Pico Interface does not support sample prediction accuracy, so
        if sample_predictions.shape != aged_samples_y_t.shape:
            sample_average_error = float('nan')
        else:
            sample_average_error = torch.norm(
                sample_predictions - aged_samples_y_t, dim=1).mean().item()

        # Test accuracy at this timestep
        decay_scalars = relative_decay[i, :]
        predictions = None
        if isinstance(model, MLPAntiDegradation) or isinstance(model, RFAntiDegradation):
            predictions = model.predict(test_X * decay_scalars, eval=True, alpha= alpha, age = age  *args.timestep, temperature = temperature[age], cleaned = cleaned[age])
        else:
            predictions = model.predict(test_X * decay_scalars, eval=True)
        average_error = torch.norm(predictions - test_y, dim=1).mean().item()
        bar.set_description(
            f"Average error at {t} hours: {average_error:.2f} mm, sample error: {sample_average_error:.2f} mm")
        errors.append(average_error)
        age += 1

    if not args.save:
        return

    avg_decay = relative_decay.mean(dim=1).cpu().numpy()
    min_decay = relative_decay.min(dim=1).values.cpu().numpy()
    max_decay = relative_decay.max(dim=1).values.cpu().numpy()

    now = int(time())

    os.makedirs(f"./saved_timeseries_runs_degradated/{task}-{now}", exist_ok=True)

    # Save the results
    _, axs = plt.subplots(3, figsize=(10, 15))

    # Plot the errors and cumulative errors
    axs[0].plot(timesteps, errors, label=f"{type(model).__name__} (Current Run)")
    axs[0].set(xlabel='Time in hours passed', ylabel='Positioning Error (mm)')
    axs[0].set(title='Positioning Error vs. LED Degradation')

    axs[1].plot(timesteps, np.cumsum(errors), label=f"{type(model).__name__} (Current Run)")
    axs[1].set(xlabel='Time in hours passed', ylabel='Cumulative Positioning Error (mm)')
    axs[1].set(title='Cumulative Positioning Error vs. LED Degradation')
    #axs[0].set_yscale("log")
    if args.compare_to:
        # Compare to past runs
        for run in args.compare_to:
            run_results = json.load(open(f"{run}/results.json", "r"))
            run_errors = run_results["errors"]
            if len(run_errors) != len(errors):
                print(f"Skipping {run} due to mismatched length of errors")
                continue
            axs[0].plot(timesteps, run_errors, label=f"{run_results["model"]} (Run {os.path.basename(run)})")
            axs[1].plot(timesteps, np.cumsum(run_errors), label=f"{run_results["model"]} (Run {os.path.basename(run)})")

    axs[0].legend()
    axs[1].legend()

    # Plot the decay scalars
    axs[2].plot(timesteps, avg_decay, label="Average Decay")
    axs[2].set(xlabel='Time in hours passed', ylabel='Average Decay Scalar')
    axs[2].set(title='Average Decay Scalar vs. Time')
    axs[2].fill_between(timesteps, min_decay, max_decay, alpha=0.2, label="Min/Max Decay")
    axs[2].legend()

    plt.savefig(f"./saved_timeseries_runs_degradated/{task}-{now}/graph.png")

    results = {
        "task": task,
        "src": args.src,
        "model": type(model).__name__,
        "errors": errors,
        "timesteps": timesteps.tolist(),
        "decay_ks": decay_ks.tolist(),
        "avg_decay": avg_decay.tolist(),
        "min_decay": min_decay.tolist(),
        "max_decay": max_decay.tolist(),
        "args": vars(args),
        "total_time": bar.format_dict['elapsed']
    }

    with open(f"./saved_timeseries_runs_degradated/{task}-{now}/results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Runner")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--src", type=str, required=True, help="Dataset name")
    parser.add_argument("--save", type=bool, required=False, help="Whether to save the run details", default=True)
    parser.add_argument("--compare-to", type=str, required=False, nargs='*',
                        help="Files of past runs to compare against", default=None)
    parser.add_argument("--load_model", type=str,
                        required=True, help="Model to load")
    parser.add_argument(
        "--std", help="Standard deviation of noise", type=float, default=0.005
    )
    parser.add_argument(
        "--time", help="Time in hours to age LEDs", type=float, default=50000)
    parser.add_argument("--samples_per_timestep",
                        help="Number of noisy samples per LED per timestep", type=int, default=100)
    parser.add_argument("--timestep", help="Timestep size",
                        type=int, required=True, default=1000)
    parser.add_argument("--flickering_prob", help="Probability of flickering",
                        type=float, required=False, default=0.001)
    parser.add_argument("--device", type=str, required=False,
                        help="Device to use", default="cpu")
    parser.add_argument("--seed", type=int, required=False,
                        help="Seed for randomness", default=42)
    parser.add_argument("--simulate_dirt_cleaning",  action=argparse.BooleanOptionalAction, help="Whether or not to simulate cleaning dirts of the sensor", default=True)
    parser.add_argument("--simulate_LED_Dirt",   action=argparse.BooleanOptionalAction,
                        help="Whether or not to simulate dirt accululating on LED's", default=True)
    parser.add_argument("--simulate_thermal_droop",  action=argparse.BooleanOptionalAction,
                        help="Whether or not to thermal droop", default=True)
    parser.add_argument("--simulate_broken_led",   action=argparse.BooleanOptionalAction,
                        help="Whether or not to simulate broken LED", default=True)
    parser.add_argument("--broken_LED_amount", type=int, required=False, default= 1)
    parser.add_argument("--specified_broken_led_times", type = int, nargs="+", required=False, default=None)


    parser.add_argument("--simulate_parttime_blockage",  action=argparse.BooleanOptionalAction,
                        help="Whether or not to simulate dirt accululating on LED's", default=True)
    parser.add_argument("--blockage_amount", type=int, required=False, default=1)
    parser.add_argument("--simulate_partitional_blockage",  action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--simulateAging",  action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    main(args)
