
"""
RANSAC for 2d lines.

"""

import numpy as np
import matplotlib.pyplot as plt
from copy import copy

from line import Line

class RANSAC:
    def __init__(self, n=10, k=100, t=0.05, d=20, model=None, loss=None, metric=None):
        self.n = n              # `n`: Минимальное количество точек данных для оценки (estimate) параметров
        self.k = k              # `k`: Допустимое максимальное количество итераций
        self.t = t              # `t`: Пороговое значение (threshold value) для определения хорошо ли подходят точки
        self.d = d              # `d`: Количество близких точек данных, необходимых для утверждения, что модель подобрана хорошо
        self.model = model      # `model`: Класс, реализующий методы `fit` и `predict`
        self.loss = loss        # `loss`: функция потерь, которая возвращает вектор
        self.metric = metric    # `metric`: функция ошибки, которая возвращает вещественное число
        self.best_fit = None
        self.best_error = np.inf

    def fit(self, X, y):

        """
        Random sample consensus (RANSAC) — это итерационный метод оценки параметров математической модели на основе 
        набора наблюдаемых данных, который содержит выбросы, но выбросы не должны влиять на значения оценок.


        Алгоритм RANSAC для подбора линейной модели к точкам данных:

        1. Выберите случайное подмножество исходных данных. Назовите это подмножество 
           гипотетическими входными значениями (inliers).

        2. Модель подгоняется к набору гипотетических входных значений (inliers).

        3. Затем все данные проверяются на соответствие подобранной модели. 
           Все точки данных (исходных данных), которые хорошо соответствуют оценочной 
           модели (estimated model) в соответствии с некоторой специфичной для модели функцией потерь, 
           называются согласованным множеством (consensus set) (т.е. набором входных значений (inliers) для модели).

        4. Оценочная модель (estimated model) достаточно хороша, если достаточно много точек данных 
           были классифицированы как часть согласованного множества (consensus set).

        5. Модель можно улучшить, переоценив ее, используя все члены согласованного множества. 
           Качество подгонки (fitting quality) как мера того, насколько хорошо модель соответствует согласованному набору, 
           будет использоваться для уточнения подгонки модели по мере продолжения итераций (например, путем 
           установки этой меры в качестве критерия качества подгонки (fitting quality criteria) на следующей итерации).

        """


        for _ in range(self.k):

            ids = np.random.choice(range(len(X)), self.k)

            maybe_inliers = ids[: self.n]
            maybe_model = copy(self.model).fit(X[maybe_inliers], y[maybe_inliers])

            thresholded = (self.loss(y[ids][self.n :], maybe_model.predict(X[ids][self.n :])) < self.t)

            inlier_ids = ids[self.n :][np.flatnonzero(thresholded).flatten()]

            if inlier_ids.size > self.d:
                inlier_points = np.hstack([maybe_inliers, inlier_ids])

                better_model = copy(self.model).fit(X[inlier_points], y[inlier_points])

                this_error = self.metric(y[inlier_points], better_model.predict(X[inlier_points]))

                if this_error < self.best_error:
                    self.best_error = this_error
                    self.best_fit = maybe_model
                    self.consensus_set = inlier_points

        return self

    def predict(self, X):
        return self.best_fit.predict(X)


    def draw(self, X, y):

        plt.style.use("seaborn-darkgrid")
        fig, ax = plt.subplots(1, 1)
        ax.set_box_aspect(1)

        outlier_points = np.array([i for i in range(0, len(X)) if i not in self.consensus_set])

        plt.plot(X[self.consensus_set], y[self.consensus_set], 'o', markersize=3, label='inlinears')
        plt.plot(X[outlier_points], y[outlier_points], 'o', markersize=3, label='outliers')

        line = np.linspace(X.min(), X.max(), num=100).reshape(-1, 1)
        plt.plot(line, self.predict(line), c="peru")
        plt.legend()
        plt.show()

def square_error_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2

def mean_square_error(y_true, y_pred):
    return np.sum(square_error_loss(y_true, y_pred)) / y_true.shape[0]