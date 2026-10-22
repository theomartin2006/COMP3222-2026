"""Lab 4, Task 1: time series data, distances and nearest neighbours (student starter)."""
import math

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from sklearn.base import BaseEstimator, ClassifierMixin

from aeon.datasets import load_basic_motions, load_gunpoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import validate_data, check_is_fitted


def plot_examples(X, y, n_per_class=3):
    """Plot the first channel of a few cases from each class."""
    for label, colour in zip(np.unique(y), ["tab:blue", "tab:red", "tab:green", "tab:orange"]):
        for series in X[y == label][:n_per_class]:
            plt.plot(series[0], color=colour, alpha=0.7)
        plt.plot([], [], color=colour, label=f"class {label}")
    plt.xlabel("Timepoint")
    plt.legend()
    plt.show()


@njit(cache=True)
def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Return the Euclidean distance between two equal-length 1D series."""
    total = 0.0
    for i in range(len(x)):
        difference = x[i] - y[i]
        total += difference * difference
    return math.sqrt(total)


@njit(cache=True)
def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Return the DTW cost between two 1D series.

    Use squared pointwise differences and do not take a square root at the end,
    so that the result matches aeon.distances.dtw_distance.
    """
    # See the Lab 1 optional extension and code/lab1/lab1_task1_solution.py.
    # C = np.full((n + 1, m + 1), np.inf); C[0, 0] = 0.0
    m = len(x)
    costs = np.full((m + 1, m + 1), np.inf)
    costs[0, 0] = 0.0

    for i in range(1, m + 1):
        for j in range(1, m + 1):
            difference = x[i - 1] - y[j - 1]
            costs[i, j] = difference**2 + min(
                costs[i - 1, j - 1], costs[i - 1, j], costs[i, j - 1]
            )
    return costs[m, m]


def check_distances():
    """Compare your distances with aeon on the pair from L4.1 slide 22."""
    from aeon.distances import dtw_distance as aeon_dtw
    from aeon.distances import euclidean_distance as aeon_euclidean

    a = np.array([1, 3, 2, 3, 15, 7, 4, 3, 1], dtype=float)
    b = np.array([2, 3, 1, 4, 4, 15, 7, 3, 1], dtype=float)
    print(f"Euclidean: yours {euclidean_distance(a, b):.3f}, aeon {aeon_euclidean(a, b):.3f}")
    print(f"DTW:       yours {dtw_distance(a, b):.3f}, aeon {aeon_dtw(a, b):.3f}")


class OneNearestNeighbour(ClassifierMixin, BaseEstimator):
    """1-NN classifier for univariate collections of shape (n_cases, 1, n_timepoints).

    distance is "euclidean" or "dtw".
    """

    def __init__(self, distance="euclidean"):
        self.distance = distance

    def fit(self, X, y):
        # TODO: Store the training series, labels and classes_.
        X = X.reshape(X.shape[0], -1)
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        # TODO: For each case in X, find the closest training case using the
        # chosen distance on channel 0 and return its label.
        check_is_fitted(self)
        X = X.reshape(X.shape[0], -1)
        X = validate_data(self, X)
        predictions = []

        for case in X:
            nearest_index = 0
            nearest_distance = np.inf
            for i, training_example in enumerate(self.X_train_):
                if self.distance == "euclidean":
                    distance = euclidean_distance(case, training_example)
                elif self.distance == "dtw":
                    distance = dtw_distance(case, training_example)

                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_index = i
            predictions.append(self.y_train_[nearest_index])

        return np.asarray(predictions, dtype=self.y_train_.dtype)


if __name__ == "__main__":
    X_train, y_train = load_gunpoint(split="train")
    X_test, y_test = load_gunpoint(split="test")
    print(f"GunPoint train: X {X_train.shape}, y {y_train.shape}")
    print(f"GunPoint test:  X {X_test.shape}, y {y_test.shape}")

    X_motion, y_motion = load_basic_motions(split="train")
    print(f"BasicMotions train: X {X_motion.shape}, classes {np.unique(y_motion)}")

    # TODO: Use plot_examples to look at both problems.
    plot_examples(X_train, y_train)
    # TODO: Implement the distances, then call check_distances().
    print()
    check_distances()

    print()
    oneNN = OneNearestNeighbour(distance="dtw").fit(X_train, y_train)
    print("Predictions:", oneNN.predict(X_test))
    print("Training accuracy:", oneNN.score(X_test, y_test))

    # TODO: Fit RandomForestClassifier on X_train[:, 0, :] and record test accuracy.

    print()
    rfc = RandomForestClassifier().fit(X_train[:, 0, :], y_train)
    print("Training accuracy for RFC:", oneNN.score(X_test, y_test))
    np.random.default_rng(0).permutation(150)

    # TODO: Shuffle the timepoints with one permutation for train and test, then repeat.
    # TODO: Compare OneNearestNeighbour with "euclidean" and "dtw", before and after
    #       shuffling, and check against KNeighborsTimeSeriesClassifier in aeon.
