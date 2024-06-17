from sklearn.datasets import make_blobs


def get_dataset(n_samples=100, random_state=None, cluster_std=1.0):
    """
    Генерирует и возвращает набор данных для обучения SVM.

    Аргументы:
    n_samples -- количество образцов (по умолчанию 100)
    random_state -- начальное состояние генератора случайных чисел (по умолчанию None)
    cluster_std -- стандартное отклонение кластеров (по умолчанию 1.0)

    Возвращает:
    X -- матрица признаков (размерность: n_samples x n_features)
    y -- вектор меток классов (размерность: n_samples)
    """
    X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=cluster_std, random_state=random_state)
    y = 2 * y - 1  # Преобразуем метки из {0, 1} в {-1, 1}
    return X, y
