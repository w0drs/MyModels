import numpy as np

class MyLogRegression:
    """
        Реализация логистической регрессии с SGD

        Параметры:
        ----------
        learning_rate : float, default=0.08
            Скорость обучения.
        n_iter : int, default=2500
            Количество итераций обучения.
    """
    def __init__(self, learning_rate=0.08, n_iter=2500):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.w = None
        self.loss_history = []
        self.acc_history = []

    def sigmoid(self, z):
        """
        Сигмоидная функция
        :param z: X * w
        """
        return 1 / (1 + np.exp(-np.clip(z, -100, 100)))

    def fit(self, X, y):
        """
        Обучение модели на данных
        :param X: фичи
        :param y: значения
        :return:
        """
        # Инициализация весов
        self.w = np.zeros(X.shape[1])

        for i in range(self.n_iter):
            # Рандомный вектор данных для SGD
            idx = np.random.randint(0, len(X))
            x_i = X[idx]
            y_i = y[idx]

            z = np.dot(x_i, self.w)
            y_pred = self.sigmoid(z)

            # Вычисление градиента
            gradient = (y_pred - y_i) * x_i

            # Обновление весов
            self.w -= self.learning_rate * gradient

            if i % 100 == 0:
                # Вычисление потерь и точности на всей выборке
                z_all = np.dot(X, self.w)
                y_pred_all = self.sigmoid(z_all)

                # Binary cross-entropy loss
                loss = -np.mean(y * np.log(y_pred_all + 1e-8) + (1 - y) * np.log(1 - y_pred_all + 1e-8))
                self.loss_history.append(loss)

                # Accuracy
                accuracy = np.mean((y_pred_all >= 0.5) == y)
                self.acc_history.append(accuracy)

                print(f"Iter {i}: Loss = {loss:.4f}, Acc = {accuracy:.4f}")

    def predict_proba(self, X):
        """
        Выдает вероятность отнесения к классу "1"
        :param X: фичи
        """
        return self.sigmoid(np.dot(X, self.w))

    def predict(self, X, threshold=0.5):
        """
        Предсказывает класс объекта (0 или 1)
        :param X: фичи
        :param threshold: пороговое значение для вероятности
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_w(self):
        """
        Возвращает коэффициенты модели
        """
        return self.w
