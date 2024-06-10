import streamlit as st
import numpy as np
from sidebar import display_data_settings, display_prediction
from visualization import plot_svm_data

# Устанавливаем seed для numpy
np.random.seed(42)

# Отображаем блоки настроек данных и предсказания
model = display_data_settings()
user_point = display_prediction(model)

# Визуализируем данные и разделяющую гиперплоскость
fig, ax = plot_svm_data(model['X'], model['y'], model['w'], model['b'])

# Отображаем точку предсказания на графике
ax.scatter(user_point[:, 0], user_point[:, 1], color='black', marker='x', label='Точка предсказания')
ax.legend()

# Отображаем график
st.pyplot(fig)
