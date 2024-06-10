import streamlit as st
import numpy as np
from data_utils import get_dataset
from svm import train_svm, predict
import matplotlib.pyplot as plt

# Устанавливаем seed для numpy
np.random.seed(42)

# Переменная состояния для хранения текущих данных
current_data = None

# Функция для генерации новых данных и обновления текущих
def generate_data(n_samples, cluster_std, random_state):
    X, y = get_dataset(n_samples=n_samples, random_state=random_state, cluster_std=cluster_std)
    return X, y

# Боковая панель с настройками данных (слева)
st.sidebar.subheader("Настройки данных")
n_samples = st.sidebar.slider("Количество точек", min_value=50, max_value=500, value=100, step=10)
cluster_std = st.sidebar.slider("Уровень дисперсии", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
random_state = st.sidebar.number_input("Сид", min_value=0, max_value=100000, value=st.session_state.get('random_state', np.random.randint(0, 100000)), step=1)

# Кнопка для генерации новых данных
if st.sidebar.button("Сгенерировать новые данные") or current_data is None:
    X, y = generate_data(n_samples, cluster_std, int(random_state))

    # Обучаем SVM
    C = 1.0  # Параметр регуляризации
    w, b = train_svm(X, y, C)

    # Сохраняем текущие данные
    current_data = {'X': X, 'y': y, 'w': w, 'b': b}

# Сохраняем сид для последующих сессий
st.session_state.random_state = int(random_state)

# Визуализируем данные и разделяющую гиперплоскость
fig, ax = plt.subplots()
ax.scatter(current_data['X'][:, 0], current_data['X'][:, 1], c=current_data['y'], cmap='bwr', alpha=0.7)
x_min, x_max = current_data['X'][:, 0].min() - 1, current_data['X'][:, 0].max() + 1
y_min, y_max = current_data['X'][:, 1].min() - 1, current_data['X'][:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], current_data['w']) + current_data['b']
Z = Z.reshape(xx.shape)
ax.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')
ax.set_xlabel('Признак 1')
ax.set_ylabel('Признак 2')
ax.set_title('Метод опорных векторов (SVM)')

# Боковая панель для предсказания (справа)
st.sidebar.subheader("Предсказание")
input_data = np.array([[st.sidebar.slider("Признак 1", min_value=current_data['X'][:, 0].min(), max_value=current_data['X'][:, 0].max(), step=0.1),
                         st.sidebar.slider("Признак 2", min_value=current_data['X'][:, 1].min(), max_value=current_data['X'][:, 1].max(), step=0.1)]])

# Предсказание пользовательских данных
if st.sidebar.button("Предсказать"):
    if current_data is not None:
        predicted_label = predict(input_data, current_data['w'], current_data['b'])
        st.sidebar.write("Предсказанная метка класса:", predicted_label[0])

# Отображаем график
st.pyplot(fig)
