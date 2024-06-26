import numpy as np
import pandas as pd
import streamlit as st

from ui.sidebar import display_data_settings, display_prediction
from ui.visualization import plot_svm_data

# Устанавливаем seed для numpy
np.random.seed(42)

tab1, tab2 = st.tabs(["Демонстрация работы", "Справочная информация"])

with tab1:
    # Отображаем блоки настроек данных и предсказания
    model = display_data_settings()
    user_point = display_prediction(model)

    # Визуализируем данные и разделяющую гиперплоскость
    fig, ax = plot_svm_data(model['X'], model['y'], model['w'], model['b'])

    # Отображаем точку предсказания на графике
    ax.scatter(user_point[:, 0], user_point[:, 1], color='black', marker='x', label='Точка предсказания')
    ax.legend()

    # Отображаем график
    st.subheader("Результат обучения")
    st.pyplot(fig)

    # Вывод данных обученной модели
    st.subheader("Данные обученной модели")
    st.write(f"Веса модели: {model['w']}")
    st.write(f"Байес: {model['b']}")

    # Отображение таблицы признаков и меток
    st.subheader("Признаки и метки обучающей выборки")
    data_table = np.hstack((model['X'], model['y'].reshape(-1, 1)))
    df = pd.DataFrame(data_table, columns=["Признак 1", "Признак 2", "Метка"])
    st.dataframe(df)

with tab2:
    st.header("Раздел справки")

    st.markdown("""
    ## Обзор
    Метод опорных векторов (SVM) является одним из мощных и популярных алгоритмов машинного обучения, который используется для задач классификации и регрессии. SVM работает путем нахождения оптимальной гиперплоскости, которая максимально разделяет данные двух классов. В двумерном пространстве эта гиперплоскость представлена линией.
    
    ### Основные понятия SVM:
    - **Опорные векторы**: Это ключевые точки данных, которые находятся на границе классов и определяют положение гиперплоскости.
    - **Гиперплоскость**: Линия или поверхность, которая разделяет данные на разные классы.
    - **Параметр регуляризации (C)**: Контролирует баланс между максимизацией ширины разделяющей гиперплоскости и минимизацией ошибки классификации.
    
    ### Применение SVM:
    Метод опорных векторов применяется в различных областях, таких как:
    - Распознавание изображений и текстов
    - Диагностика заболеваний
    - Анализ финансовых данных
    - Обработка естественного языка
    
    SVM особенно полезен в случаях, когда необходимо провести классификацию данных с большим количеством признаков и четкой границей между классами.
    
    ## Функционал
    Вкладка **Главная** позволяет настроить параметры для генерации данных и классификации:
    - **Количество точек**: Регулирует количество генерируемых точек данных.
    - **Уровень дисперсии**: Устанавливает уровень шума в данных.
    - **Параметр регуляризации (C)**: Контролирует строгость разделяющей гиперплоскости. Низкие значения увеличивают допуск к ошибкам, высокие значения уменьшают допуск.
    - **Сид**: Устанавливает начальное значение для генератора случайных чисел, обеспечивая воспроизводимость данных.
    
    Нажатие на кнопку "Сгенерировать новые данные" обновит сид на случайное число и сгенерирует данные, которые затем будут классифицированы с использованием SVM на основе указанных параметров.
    
    ## Визуализация
    Генерируемые данные и результаты классификации отображаются на графике. На графике:
    - Точки данных представлены цветными маркерами: один цвет для каждого класса.
    - Разделяющая гиперплоскость, найденная SVM, отображается в виде линии.
    - Линии, обозначающие границы классов, отображаются пунктиром.
    
    Для визуализации используется следующая логика:
    - Точки данных (X и y) отображаются на графике.
    - С помощью SVM вычисляется разделяющая гиперплоскость.
    - Отображаются линии, соответствующие границам классов (-1, 0, 1).
    
    ## Предсказание точки
    В этом приложении также есть функционал предсказания класса для новой точки. Вы можете использовать соответствующие ползунки для задания значений признаков новой точки, и приложение предскажет, к какому классу она принадлежит:
    - **Признак 1** и **Признак 2**: Устанавливают значения признаков новой точки.
    - Нажмите кнопку "Предсказать" для получения предсказания класса, который будет отображен как "синий (-1)" или "красный (1)".
    
    Вкладка **Справка** предоставляет подробный обзор метода опорных векторов, его применение и функциональные возможности, доступные на вкладке **Главная**.
     """)
