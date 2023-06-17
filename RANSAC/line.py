"""
Line module.

"""

import numpy as np

class Line():
    def __init__(self):
        self.params = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Подгонка линейной модели к данному набору гипотетических входных значений (inliers)

        """

        A = np.vstack([X.flatten(), np.ones(len(X))]).T
        self.m, self.c = np.linalg.lstsq(A, y.flatten(), rcond=None)[0]

        return self

    def predict(self, X: np.ndarray):
        return self.m * X + self.c
