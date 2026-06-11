import argparse
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from dataset import FPDataset


def export_dataset(csv_path, device, x_out, y_out):
    dataset = FPDataset(csv_path, device)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    X, y = next(iter(loader))

    X = X.detach().cpu().numpy().astype(np.float32)
    y = y.detach().cpu().numpy().astype(np.float32)

    print(csv_path)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    np.save(x_out, X)
    np.save(y_out, y)

    print(f"Saved {x_out}")
    print(f"Saved {y_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    export_dataset(
        csv_path=args.dataset + "/train.csv",
        device=args.device,
        x_out="X_train.npy",
        y_out="y_train.npy",
    )

    export_dataset(
        csv_path=args.dataset + "/test.csv",
        device=args.device,
        x_out="X_test.npy",
        y_out="y_test.npy",
    )