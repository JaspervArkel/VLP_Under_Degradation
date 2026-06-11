import pickle

import torch
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader

from dataset import FPDataset
from models.base import BaseModel

lastCleaned = 0

def antiAge(X, age, alpha,):
    #todo
    X = X * (1 /(1-alpha/(365 * 24)))**age
    return X


def antiDust(X,cleaned,time):

    global lastCleaned
    if cleaned:
        lastCleaned = time

    dustcollectiontime = time - lastCleaned
    expectedreductiononsensor = (1- 0.015/(730*24)) **dustcollectiontime #between 1-2.5 in 730 days
    expectedreductiononlights = (1-0.08/(730*24)) **dustcollectiontime
    X = X*(1/expectedreductiononlights) *(1/expectedreductiononsensor)
    return X


def antiTermalDroop(X ,temp, coefficient):
    change =(1/(1-coefficient*temp))
    X= X* change
    return X


def DropDetection(X: torch.Tensor):
    global dropped
    # for x in X:
    #     if x < 0:
    #         return X
    return X


def antiDegrade(X,temperature, age = None, alpha = None, cleaned = False,  thermal_coeff = 0.003):

    if age is not None:
        X = antiAge(X, age, alpha)
        X = antiDust(X,cleaned,age)
    X = antiTermalDroop(X, temperature, thermal_coeff)
    X = DropDetection(X)

    return X


class RFAntiDegradation(BaseModel):
    def __init__(self, n_estimators=100, max_depth=None, device="cpu", seed=None):
        """
        Initialize the Random Forest model.

        :param n_estimators: The number of trees in the forest.
        :param max_depth: The maximum depth of the tree.
        :param seed: Controls the randomness of the estimator.
        """
        self.device = device
        self.train_X = None
        self.train_y = None
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, verbose=2, n_jobs=-1, random_state=seed
        )

    def fit(self, dataset: FPDataset):
        """
        Fit the model to the dataset.

        :param dataset: The dataset to fit the model to.
        """

        loader = DataLoader(dataset, batch_size=len(dataset))
        AllX = []
        AllY = []
        for X_ref, y_ref in loader:
            AllX.append(X_ref.detach().cpu())
            AllY.append(y_ref.detach().cpu())

        self.train_X = torch.cat(AllX, dim=0).float().to(self.device)
        self.train_y = torch.cat(AllY, dim=0).float().to(self.device)

        max_knn_samples = 1000
        if self.train_X.shape[0] > max_knn_samples:
            idx = torch.randperm(self.train_X.shape[0], device=self.device)[:max_knn_samples]
            self.train_X = self.train_X[idx]
            self.train_y = self.train_y[idx]
        print("Stored kNN reference data:", self.train_X.shape, self.train_y.shape)


        X_tensor, y_tensor = next(iter(loader))
        X = X_tensor.numpy()
        y = y_tensor.numpy()

        self.model.fit(X, y)

    def repareBockageandBroken(self, X):


        with torch.no_grad():
            roughPos = self.model.predict(X.numpy())
            roughPos = torch.as_tensor(
                roughPos,
                dtype=self.train_y.dtype,
                device=self.train_y.device
            )

            distsances = torch.cdist(roughPos, self.train_y)


            knn_dist, knn_idx = torch.topk(distsances, k=5, largest=False)


            neighbour_leds = self.train_X[knn_idx]


            weights = 1.0 / (knn_dist + 1e-8)
            weights = weights / weights.sum(dim=1, keepdim=True)

            expected = (neighbour_leds * weights.unsqueeze(-1)).sum(dim=1)


            diffratio = X / (expected + 1e-8)
            suspicious = (expected > 1e-4) & (diffratio < 0.7)

            print(f"Replaced {suspicious.sum().item()} LED values")

            X = X.clone()
            X[suspicious] = expected[suspicious]


        return X

    def predict(self, X: torch.Tensor, eval: bool = False, **kwargs) -> torch.Tensor:
        """
        Predict using the model on the dataset.

        :param X: The data to predict on.
        :param eval: Whether or not it's in evaluation mode.
        :return: The predictions.
        """

        X = antiDegrade(X,kwargs.get("temperature"),alpha= kwargs.get("alpha",0.03), age= kwargs.get("age",0 ),cleaned = kwargs.get("cleaned", False),
                         )

        X = self.repareBockageandBroken(X)
        return torch.from_numpy(self.model.predict(X.numpy()))

    def save(self, model_path: str):
        """
        Save the model to the specified path using pickle.

        :param model_path: The path to save the model.
        """
        model_path = model_path + ".pickle"
        checkpoint = {
            "model": self.model,
            "train_X": self.train_X,
            "train_y": self.train_y,
        }

        with open(model_path, "wb") as f:
            pickle.dump(checkpoint, f)

        return model_path

    def load(self, model_path: str):
        """
        Load the model from the specified path using pickle.

        :param model_path: The path to load the model from.
        :return: The loaded model.
        """
        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)

        if isinstance(checkpoint, dict):
            self.model = checkpoint["model"]
            self.train_X = checkpoint.get("train_X", None)
            self.train_y = checkpoint.get("train_y", None)



        if self.train_X is not None:
            self.train_X = self.train_X.to(self.device)

        if self.train_y is not None:
            self.train_y = self.train_y.to(self.device)

        return
        raise ValueError(f"Model at {model_path} not loaded")
