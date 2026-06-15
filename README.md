# VLP Under Degradation


This repository contains the code and experiments for **<brief description of your study>**.

The implementation used in this study is based on the original [VLP study by Joey Wenyi Li](https://github.com/joeywli/VLP.git).

The original repository contains the complete implementation and documentation for:

* Data preparation
* Heatmap generation
* Data cleaning
* LED aging simulation
* Model training
* Time-series experiments

This README only provides a brief overview of the required preparation. For detailed instructions, please refer to the [original repository](https://github.com/JaspervArkel/VLP_Under_Degradation).

---

## Preparation

### 1. Clone this repository

```bash
git clone <URL-OF-THIS-REPOSITORY>
cd <REPOSITORY-NAME>
```

### 2. Create a virtual environment

Python 3.13 was used for the original implementation.

```bash
python -m venv .venv
```

Activate the virtual environment.

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the RANSAC line-fitting dependency

The original implementation uses the [ransac-line](https://github.com/einstein8612/ransac-line) library.

```bash
mkdir third_party
cd third_party

git clone https://github.com/einstein8612/ransac-line.git
cd ransac-line/bindings/python/ransac_line

pip install cffi setuptools
python build_ffi.py

cd ..
pip install .

cd ../../../..
```

### 5. Prepare the dataset

Place the original MATLAB files in:

```text
dataset/mat_files/
```

Convert the MATLAB files to the dataset format used by the experiments:

```bash
python dataset/convert.py \
  --src "./dataset/mat_files" \
  --dst "./dataset/exported" \
  --normalise true \
  --training_fraction 0.8 \
  --seed 42
```

### 6. Generate the heatmaps

```bash
python dataset/heatmap.py \
  --src "./dataset/exported/data.csv" \
  --dst "./dataset/heatmaps" \
  --imgs true
```

## Training a Model

Model training is performed using `experiment.py`.

Before starting, make sure that the selected dataset directory contains:

```text
<dataset>/
├── train.csv
└── test.csv
```

Use the following general command:

```bash
python experiment.py \
  --task <TASK> \
  --dataset <DATASET_DIRECTORY> \
  --seed 42
```

The `--dataset` argument must point to the directory containing `train.csv` and `test.csv`.

### Example: Random Forest

The following command trains the small Random Forest configuration used in the original repository:

```bash
python experiment.py \
  --task "RF-TINY" \
  --dataset "./dataset/exported/data_176" \
  --seed 42
```

After training, the model is saved automatically in:

```text
saved_runs/
```

For example:

```text
saved_runs/RF-TINY-1745593383.pickle
```



### Available model tasks

The model configurations are registered in `models/__init__.py`. Available task names include:

```text
RF
RF-TINY
MLP
MLP-TINY
MLP-TINY-NORMALISE
MLP-ONLINE-TINY
MLP-ONLINE-PICO
KNN
WKNN
WOKNN
WOKNN-ONLINE
RESIDUAL-MLP-ONLINE
RESIDUAL-MLP-ONLINE-SPARSE
MLP-ANTI-DEGRADING
RF-ANTI-DEGRADING
MLP-ANTI-DEGRADING-PICO
RF-ANTI-DEGRADING-PICO
PICO-INTERFACE
```

Additional preparation steps, including heatmap cleaning, LED position estimation, LED degradation simulation, downsampling, augmentation and model training, are described in the [original VLP Under Degradation repository](https://github.com/JaspervArkel/VLP_Under_Degradation).

---

## Run the Degradation Time-Series Experiment

The extended degradation experiment is implemented in:

```text
Experiment_timeseries_degradation.py
```

This experiment evaluates a trained positioning model while simulating several degradation effects over time, including:

* LED ageing
* Dirt accumulation
* Dirt cleaning
* Thermal droop
* Broken LEDs
* Temporary blockages
* Partial blockages
* Measurement flickering

Before running the experiment, make sure that you have:

* A trained model in `saved_runs/`
* A cleaned heatmap stored as a `.npy` file
* Installed the required Python dependencies

### Basic command

```bash
python Experiment_timeseries_degradation.py \
  --task "<TASK>" \
  --load_model "./saved_runs/<MODEL_FILE>" \
  --src "./dataset/heatmaps/heatmap_176/cleaned_LAMBERTIAN-IDW.npy" \
  --timestep 1000 \
  --time 100000 \
  --seed 42
```

For example:

```bash
python Experiment_timeseries_degradation.py \
  --task "MLP-TINY" \
  --load_model "./saved_runs/MLP-TINY-<TIMESTAMP>.pth" \
  --src "./dataset/heatmaps/heatmap_176/cleaned_LAMBERTIAN-IDW.npy" \
  --timestep 1000 \
  --time 100000 \
  --seed 42
```

The value supplied to `--task` must match the type of the loaded model.

### Main parameters

| Parameter                | Description                                    |  Default |
| ------------------------ | ---------------------------------------------- | -------: |
| `--task`                 | Registered model configuration                 | Required |
| `--load_model`           | Path to the trained model                      | Required |
| `--src`                  | Path to the cleaned heatmap `.npy` file        | Required |
| `--timestep`             | Number of simulated hours between evaluations  | Required |
| `--time`                 | Total simulated time in hours                  |  `50000` |
| `--samples_per_timestep` | Number of generated samples at each timestep   |    `100` |
| `--std`                  | Standard deviation of degradation noise        |  `0.005` |
| `--flickering_prob`      | Probability of measurement flickering          |  `0.001` |
| `--device`               | Device used for model execution                |    `cpu` |
| `--seed`                 | Random seed                                    |     `42` |
| `--broken_LED_amount`    | Number of LEDs that fail during the experiment |      `1` |
| `--blockage_amount`      | Number of simulated temporary blockages        |      `1` |

### Select degradation effects

All degradation effects are enabled by default. Individual effects can be disabled using their corresponding `--no-...` argument.

For example, run the experiment without thermal droop and without broken LEDs:

```bash
python Experiment_timeseries_degradation.py \
  --task "MLP-TINY" \
  --load_model "./saved_runs/MLP-TINY-<TIMESTAMP>.pth" \
  --src "./dataset/heatmaps/heatmap_176/cleaned_LAMBERTIAN-IDW.npy" \
  --timestep 1000 \
  --time 100000 \
  --no-simulate_thermal_droop \
  --no-simulate_broken_led \
  --seed 42
```

The available degradation options are:

| Enabled argument                  | Disabled argument                    | Simulated effect                  |
| --------------------------------- | ------------------------------------ | --------------------------------- |
| `--simulateAging`                 | `--no-simulateAging`                 | Gradual LED ageing                |
| `--simulate_LED_Dirt`             | `--no-simulate_LED_Dirt`             | Dirt accumulation on LEDs         |
| `--simulate_dirt_cleaning`        | `--no-simulate_dirt_cleaning`        | Periodic dirt cleaning            |
| `--simulate_thermal_droop`        | `--no-simulate_thermal_droop`        | Temperature-related thermal droop |
| `--simulate_broken_led`           | `--no-simulate_broken_led`           | Complete LED failure              |
| `--simulate_parttime_blockage`    | `--no-simulate_parttime_blockage`    | Temporary LED blockage            |
| `--simulate_partitional_blockage` | `--no-simulate_partitional_blockage` | Partial LED blockage              |

### Example with multiple broken LEDs

```bash
python Experiment_timeseries_degradation.py \
  --task "MLP-TINY" \
  --load_model "./saved_runs/MLP-TINY-<TIMESTAMP>.pth" \
  --src "./dataset/heatmaps/heatmap_176/cleaned_LAMBERTIAN-IDW.npy" \
  --timestep 1000 \
  --time 100000 \
  --broken_LED_amount 3 \
  --blockage_amount 2 \
  --seed 42
```

### GPU execution

A PyTorch model can be evaluated on a CUDA-compatible GPU using:

```bash
python Experiment_timeseries_degradation.py \
  --task "MLP-TINY" \
  --load_model "./saved_runs/MLP-TINY-<TIMESTAMP>.pth" \
  --src "./dataset/heatmaps/heatmap_176/cleaned_LAMBERTIAN-IDW.npy" \
  --timestep 1000 \
  --time 100000 \
  --device "cuda:0" \
  --seed 42
```

### Output

When saving is enabled, the results are stored automatically in:

```text
saved_timeseries_runs_degradated/<TASK>-<TIMESTAMP>/
```

Each run contains:

```text
saved_timeseries_runs_degradated/
└── <TASK>-<TIMESTAMP>/
    ├── graph.png
    └── results.json
```

The generated graph shows:

1. Positioning error over time
2. Cumulative positioning error over time
3. Average LED decay, including minimum and maximum decay values

The `results.json` file contains the experiment configuration, timesteps, positioning errors and simulated decay values.


## Original implementation

This study builds upon the implementation provided in:

* [joeywli\VLP](https://github.com/joeywli/VLP.git)

Please refer to the original repository for the complete preparation pipeline, original experiments and detailed documentation.

## Citation

When using or extending  this implementation, please cite or reference:

```text
VLP Under Degradation
https://github.com/JaspervArkel/VLP_Under_Degradation
```

Add the formal citation for the associated paper or thesis here when applicable:

```bibtex
@misc{vlp_under_degradation,
  author       = {<Jasper van Arkel>},
  title        = {VLP Under Degradation},
  year         = {<2026>},
  howpublished = {\url{https://github.com/JaspervArkel/VLP_Under_Degradation}}
}
```

## License

<Add the license used for this repository.>

Parts of this project may be based on code from the original VLP Under Degradation repository. Verify the license of the original repository and preserve all required copyright and license notices.
