import matplotlib.pyplot as plt
import numpy as np


def plot_svm_data(X, y, w, b):
    """
    Функция для визуализации данных и разделяющей гиперплоскости метода опорных векторов (SVM).

    Parameters:
        X (array-like): Массив признаков.
        y (array-like): Массив меток классов.
        w (array-like): Вектор весов.
        b (float): Смещение.
    """
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    Z = Z.reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title('Метод опорных векторов (SVM)')
    return fig, ax
