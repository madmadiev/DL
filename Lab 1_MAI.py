import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

class SingleLayerPerceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000, error_threshold=0.001):
        #Инициализация перцептрона
        self.weights = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.error_threshold = error_threshold
        self.input_size = input_size
        
    def activation_function(self, x):
        #Сигмоидная функция активации
        return 1 / (1 + np.exp(-x))
    
    def activation_function_derivative(self, x):
        return self.activation_function(x) * (1 - self.activation_function(x))
    
    def init_weights(self):
        self.weights = np.random.randn(self.input_size + 1) * 0.01
        
    def predict(self, X):
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        net = np.dot(X_with_bias, self.weights)
        y = self.activation_function(net)
        return (y >= 0.5).astype(int)
    
    def train(self, X, y):
        #Обучение перцептрона
        self.init_weights()

        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        
        for epoch in range(self.epochs):
            total_error = 0
            delta_weights = np.zeros_like(self.weights)
            
            for i in range(X_with_bias.shape[0]):
                x_i = X_with_bias[i]
                d_i = y[i]
                net = np.dot(x_i, self.weights)
                y_out = self.activation_function(net)
                error = 0.5 * (d_i - y_out)**2
                total_error += error
                delta = -(d_i - y_out) * self.activation_function_derivative(net)
                delta_weights += -self.learning_rate * delta * x_i
            
            average_error = total_error / X.shape[0]
            if epoch % 100 == 0:
                print(f"Эпоха {epoch}, Средняя ошибка: {average_error:.4f}")
                
            if average_error < self.error_threshold:
                print(f"Обучение завершено на эпохе {epoch}, ошибка достигла порога")
                break
            self.weights += delta_weights
        
        print("Обучение завершено")

# Загрузка данных
data = load_breast_cancer()
X = data.data
y = data.target

# Нормализация данных
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение перцептрона
perceptron = SingleLayerPerceptron(input_size=X.shape[1], learning_rate=0.01, epochs=1000)
perceptron.train(X_train, y_train)

# Вывод точности
y_pred = perceptron.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.4f}")
