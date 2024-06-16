import numpy as np
import matplotlib.pyplot as plt
from data_utils import get_dataset
from algorithms.svm import train_svm, predict


def plot_svm(X, y, w, b):
    """
    Визуализирует данные и разделяющую гиперплоскость.

    Аргументы:
    X -- матрица признаков (размерность: n_samples x n_features)
    y -- вектор меток классов (размерность: n_samples)
    w -- вектор весов (размерность: n_features)
    b -- смещение (скаляр)
    """
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)

    # Создаем сетку для построения разделяющей гиперплоскости
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    # Вычисляем значения для гиперплоскости
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    Z = Z.reshape(xx.shape)

    # Рисуем разделяющую гиперплоскость и маргинальные линии
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.title('Метод опорных векторов (SVM)')
    plt.show()


if __name__ == "__main__":
    # Получаем данные с различными параметрами
    X, y = get_dataset(n_samples=150, random_state=None, cluster_std=1.5)

    # Обучаем SVM
    C = 1.0  # Параметр регуляризации
    w, b = train_svm(X, y, C)

    # Выводим результаты
    print("Вектор весов:", w)
    print("Смещение:", b)

    # Визуализируем данные и разделяющую гиперплоскость
    plot_svm(X, y, w, b)

    # Пример предсказания
    X_test = np.random.randn(5, 2)
    y_pred = predict(X_test, w, b)
    print("Предсказанные метки классов:", y_pred)
