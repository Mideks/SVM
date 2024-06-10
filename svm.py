import numpy as np
from interior_point_method import kernel, solve_qp


def train_svm(X, y, C):
    """
    Обучает SVM используя метод внутренней точки.

    Аргументы:
    X -- матрица признаков (размерность: n_samples x n_features)
    y -- вектор меток классов (размерность: n_samples)
    C -- параметр регуляризации

    Возвращает:
    w -- вектор весов (размерность: n_features)
    b -- смещение (скаляр)
    """
    n_samples, n_features = X.shape

    # Вычисляем матрицу ядра
    K = np.array([[kernel(X[i], X[j]) for j in range(n_samples)] for i in range(n_samples)])

    # Решаем задачу квадратичного программирования
    alphas, b = solve_qp(K, y, C)

    # Вычисляем вектор весов
    w = np.dot((alphas * y).T, X)
    return w, b


def predict(X, w, b):
    """
    Предсказывает метки классов для данных X.

    Аргументы:
    X -- матрица признаков (размерность: n_samples x n_features)
    w -- вектор весов (размерность: n_features)
    b -- смещение (скаляр)

    Возвращает:
    y_pred -- предсказанные метки классов (размерность: n_samples)
    """
    return np.sign(np.dot(X, w) + b)
