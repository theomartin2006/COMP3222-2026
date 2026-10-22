import pandas as pd
from pathlib import Path
import numpy as np

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "lab2" / "playgolf.csv"


def impurity(counts: np.ndarray) -> float:
    """Return Gini impurity for one node given class counts"""
    sum = 0
    total = np.sum(counts)
    for count in counts:
        sum += (count / total) ** 2

    return 1 - sum

def gini_gain(attr: np.ndarray, y: np.ndarray) -> float:
    """Compute Gini gain for splitting labels y by vategorical attribute attr"""



if __name__ == "__main__":  # Example usage
    data = pd.read_csv(DATA_PATH)
    print(data)
    y = data.iloc[:, 4].to_numpy()
    print(y)
    X = data.iloc[:, 0:4].to_numpy()
    print(X.shape)
    outlook = X[:, 0]
    print(outlook)
    print(outlook.shape)
    print(impurity(np.unique(y, return_counts=True)[1]))
