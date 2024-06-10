import numpy as np


def kernel(x1, x2):
    """Линейное ядро."""
    return np.dot(x1, x2)


def solve_qp(K, y, C, tol=1e-4, max_passes=5):
    """
    Решает задачу квадратичного программирования для SVM с помощью метода внутренней точки.

    Аргументы:
    K -- матрица ядра (размерность: n_samples x n_samples)
    y -- вектор меток классов (размерность: n_samples)
    C -- параметр регуляризации
    tol -- допустимая погрешность
    max_passes -- максимальное количество проходов по данным без изменений

    Возвращает:
    alphas -- оптимальные значения альфа (размерность: n_samples)
    b -- смещение (скаляр)
    """
    n_samples = len(y)
    alphas = np.zeros(n_samples)
    b = 0
    passes = 0

    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(n_samples):
            Ei = np.dot(alphas * y, K[:, i]) + b - y[i]
            if (y[i] * Ei < -tol and alphas[i] < C) or (y[i] * Ei > tol and alphas[i] > 0):
                j = np.random.choice([x for x in range(n_samples) if x != i])
                Ej = np.dot(alphas * y, K[:, j]) + b - y[j]

                alpha_i_old, alpha_j_old = alphas[i], alphas[j]

                if y[i] != y[j]:
                    L, H = max(0, alphas[j] - alphas[i]), min(C, C + alphas[j] - alphas[i])
                else:
                    L, H = max(0, alphas[i] + alphas[j] - C), min(C, alphas[i] + alphas[j])

                if L == H:
                    continue

                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                alphas[j] -= y[j] * (Ei - Ej) / eta
                alphas[j] = np.clip(alphas[j], L, H)

                if abs(alphas[j] - alpha_j_old) < tol:
                    continue

                alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])

                b1 = b - Ei - y[i] * (alphas[i] - alpha_i_old) * K[i, i] - y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                b2 = b - Ej - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - y[j] * (alphas[j] - alpha_j_old) * K[j, j]

                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                num_changed_alphas += 1

        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0

    return alphas, b
