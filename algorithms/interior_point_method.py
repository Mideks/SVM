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
    n_samples = len(y)  # Количество образцов
    alphas = np.zeros(n_samples)  # Инициализация коэффициентов альфа нулями
    b = 0  # Инициализация смещения
    passes = 0  # Счётчик проходов без изменений

    while passes < max_passes:
        num_changed_alphas = 0  # Счётчик изменённых альф за один проход
        for i in range(n_samples):
            Ei = np.dot(alphas * y, K[:, i]) + b - y[i]  # Ошибка предсказания для i-го образца
            # Условия для обновления альфы
            if (y[i] * Ei < -tol and alphas[i] < C) or (y[i] * Ei > tol and alphas[i] > 0):
                j = np.random.choice([x for x in range(n_samples) if x != i])  # Выбор случайного j, не равного i
                Ej = np.dot(alphas * y, K[:, j]) + b - y[j]  # Ошибка предсказания для j-го образца

                alpha_i_old, alpha_j_old = alphas[i], alphas[j]  # Сохранение старых значений альфа_i и альфа_j

                # Вычисление границ L и H
                if y[i] != y[j]:
                    L, H = max(0, alphas[j] - alphas[i]), min(C, C + alphas[j] - alphas[i])
                else:
                    L, H = max(0, alphas[i] + alphas[j] - C), min(C, alphas[i] + alphas[j])

                if L == H:
                    continue  # Пропуск, если L == H

                eta = 2 * K[i, j] - K[i, i] - K[j, j]  # Вычисление eta
                if eta >= 0:
                    continue  # Пропуск, если eta >= 0

                # Обновление альфа_j
                alphas[j] -= y[j] * (Ei - Ej) / eta
                alphas[j] = np.clip(alphas[j], L, H)  # Ограничение альфа_j в пределах [L, H]

                if abs(alphas[j] - alpha_j_old) < tol:
                    continue  # Пропуск, если изменение альфа_j недостаточно

                # Обновление альфа_i
                alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])

                # Вычисление смещений b1 и b2
                b1 = b - Ei - y[i] * (alphas[i] - alpha_i_old) * K[i, i] - y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                b2 = b - Ej - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - y[j] * (alphas[j] - alpha_j_old) * K[j, j]

                # Обновление смещения b
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                num_changed_alphas += 1  # Увеличение счётчика изменённых альф

        if num_changed_alphas == 0:
            passes += 1  # Увеличение счётчика проходов без изменений
        else:
            passes = 0  # Сброс счётчика проходов без изменений

    return alphas, b  # Возвращение оптимальных значений альфа и смещения b
