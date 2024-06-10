import random

import numpy as np
import streamlit as st

from data_utils import get_dataset
from svm import train_svm, predict


def generate_data(n_samples, cluster_std, random_state):
    X, y = get_dataset(n_samples=n_samples, random_state=random_state, cluster_std=cluster_std)
    return X, y


def display_data_settings(model):
    with st.sidebar.expander("Настройки данных", expanded=True):
        n_samples = st.slider("Количество точек", min_value=50, max_value=500, value=100, step=10, help="Количество точек в сгенерированных данных.")
        cluster_std = st.slider("Уровень дисперсии", min_value=0.1, max_value=5.0, value=1.0, step=0.1, help="Уровень шума в данных.")
        C = st.slider("Параметр регуляризации (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.01, help="Параметр регуляризации в методе опорных векторов (SVM). Большие значения C приводят к меньшему запрещенному нарушению маржи, что может привести к более сложной модели.")

        seed = st.session_state.get('random_state', None)
        # Кнопка для генерации новых данных
        if st.button("Сгенерировать новые данные") or seed is None:
            seed = random.randint(0, 100000)  # Генерируем новый сид
            st.session_state.random_state = seed  # Сохраняем сид в сессии

        random_state = st.number_input("Сид", min_value=0, max_value=100000, value=seed, step=1, help="Сид для генерации случайных данных.")

        X, y = generate_data(n_samples, cluster_std, int(random_state))

        # Обучаем SVM
        w, b = train_svm(X, y, C)

        # Сохраняем текущие данные
        model = {'X': X, 'y': y, 'w': w, 'b': b}

    return model


def display_prediction(model):
    with st.sidebar.expander("Предсказание", expanded=True):
        # Поле для ввода пользовательских данных
        user_point = np.array([[
            st.slider("Признак 1", min_value=model['X'][:, 0].min(), max_value=model['X'][:, 0].max(), step=0.1),
            st.slider("Признак 2", min_value=model['X'][:, 1].min(), max_value=model['X'][:, 1].max(), step=0.1)
        ]])

        # Предсказание пользовательских данных
        if st.button("Предсказать"):
            predicted_label = predict(user_point, model['w'], model['b'])
            st.write("Предсказанная метка класса:", predicted_label[0])

    return user_point
