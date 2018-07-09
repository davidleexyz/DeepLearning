import numpy as np

class LinearRegression():

	def __init__(self):
		pass

	def train_gradient_descent(self, X, y, learning_rate=0.01, n_iters=100):
		n_samples, n_features = X.shape
		self.weights = np.zeros(shape=(n_features, 1))
		self.bias = 0
		costs = []

		for i in range(n_iters):
			y_predict = np.dot(X, self.weights) + self.bias

			cost = (1 / n_samples) * np.sum((y_predict - y) ** 2)
			costs.append(cost)

			if i % 100 == 0:
				print(f"Cost at iteration {i}: {cost}")

			dJ_dw = ( 2 / n_samples ) * np.dot(X.T, (y_predict - y))
			dJ_db = ( 2 / n_samples ) * np.sum((y_predict - y))

			self.weights = self.weights - learning_rate * dJ_dw
			self.bias = self.bias - learning_rate * dJ_db

		return self.weights, self.bias, costs

	def predict(self, X):
		return np.dot(X, self.weights) + self.bias
