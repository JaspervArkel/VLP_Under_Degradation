import torch
from setuptools.config.setupcfg import AllCommandOptions
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset import FPDataset
from models.base import BaseModel

dropped = np.zeros(36)
lastCleaned = 0
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


class NormalizeInput(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=1, keepdim=True) + 1e-8)


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


def antiDegrade(X,temperature, age = None, alpha = None, cleaned = False,  thermal_coeff = 0.006):

    if age is not None:
        X = antiAge(X, age, alpha)
        X = antiDust(X,cleaned,age)
    X = antiTermalDroop(X, temperature, thermal_coeff)
    X = DropDetection(X)

    return X





class MLPAntiDegradation(BaseModel):
    def __init__(self, batch_size=64, lr=0.001, epochs=25, normalize=False, device="cpu", seed=None):
        """
        Initialize the MLP model.

        :param seed: Controls the randomness of the estimator.
        """
        self.device = device
        self.train_X = None
        self.train_y = None
        self.model = nn.Sequential(
            nn.Linear(36, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 2),
        ).to(device)

        # Normalize input if necessary
        if normalize:
            self.model.insert(0, NormalizeInput())

        self.model.apply(init_weights)

        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.seed = seed

    def repareBockageandBroken(self, X):


        with torch.no_grad():
            roughPos = self.model(X)


            distsances = torch.cdist(roughPos, self.train_y)


            knn_dist, knn_idx = torch.topk(distsances, k=5, largest=False)


            neighbour_leds = self.train_X[knn_idx]


            weights = 1.0 / (knn_dist + 1e-8)
            weights = weights / weights.sum(dim=1, keepdim=True)

            expected = (neighbour_leds * weights.unsqueeze(-1)).sum(dim=1)


            diffratio = X / (expected + 1e-8)
            suspicious = (expected > 1e-4) & (diffratio < 0.25)

            print(f"Replaced {suspicious.sum().item()} LED values")

            X = X.clone()
            X[suspicious] = expected[suspicious]


        return X

    def fit(self, dataset: FPDataset):
        """
        Fit the model to the dataset.

        :param dataset: The dataset to fit the model to.
        """
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        AllX =[]
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
        
        print("it works")
        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        torch.manual_seed(self.seed)

        # Training loop
        self.model.train()
        for _ in tqdm(range(self.epochs), desc="Training MLP", unit="epoch"):
            for X, y in loader:
                optimizer.zero_grad()
                outputs = self.model(X)
                # loss = (torch.norm(outputs - y.float(), dim=1)**2).mean()
                loss = criterion(outputs, y.float())
                loss.backward()
                optimizer.step()

    def predict(self ,X: torch.Tensor, eval = False, **kwargs ) -> torch.Tensor:
        """
        Predict using the model on the dataset.

        :param X: The data to predict on.
        :param eval: Whether or not it's in evaluation mode.
        :return: The predictions.
        """
        print("inside predict kwargs:", kwargs)
        self.model.eval()

        X = antiDegrade(X,kwargs.get("temperature"),alpha= kwargs.get("alpha",0.03), age= kwargs.get("age",0 ),cleaned = kwargs.get("cleaned", False),
                         )
        X = self.repareBockageandBroken(X)
        if eval:
            return self.model(X)



        return self.model.forward(X)

    def save(self, model_path: str):
        """
        Save the model to the specified path using pickle.

        :param model_path: The path to save the model.
        """
        model_path = model_path + ".pth"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "train_X": self.train_X,
            "train_y": self.train_y,
        }, model_path)
        return model_path

    def load(self, model_path: str):
        """
        Load the model from the specified path using pickle.

        :param model_path: The path to load the model from.
        :return: The loaded model.
        """
        with open(model_path, "rb") as f:
            checkpoint = torch.load(model_path, map_location=self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.train_X = checkpoint.get("train_X", None)
            self.train_y = checkpoint.get("train_y", None)

            if self.train_X is not None:
                self.train_X = self.train_X.to(self.device)

            if self.train_y is not None:
                self.train_y = self.train_y.to(self.device)

            print(
                "Loaded kNN reference data:",
                None if self.train_X is None else self.train_X.shape,
                None if self.train_y is None else self.train_y.shape,
            )
            return
        raise ValueError(f"Model at {model_path} not loaded")
