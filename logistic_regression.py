import numpy as np 

class LogisticRegression():

	def __init__(self):
		pass

	def sigmoid(self, a):
		return 1 / (1 + np.exp(-a))

	def train(self, x, y_true, n_iters, learning_rate):
		n_samples, n_features = x.shape
		self.weights = np.zeros((n_features, 1))
		self.bias = 0
		costs = []

		for i in range(n_iters):
			y_predict = self.sigmoid(np.dot(x, self.weights) + self.bias)

			cost = (-1 / n_samples) * np.sum(y_true * np.log(y_predict) + (1 - y_true) * (np.log(1 - y_predict)))

			dw = (1 / n_samples) * np.dot(x.T, (y_predict - y_true))
			db = (1 / n_samples) * np.sum(y_predict - y_true)

			self.weights = self.weights - learning_rate * dw
			self.bias = self.bias - learning_rate * db

			costs.append(cost)

			if i % 100  == 0:
				print(f"Cost after iteration {i}: {cost}")

		return self.weights, self.bias, costs

	def predict(self, x):
		y_predict = self.sigmoid(np.dot(x, self.weights) + self.bias)
		y_predict_labels = [1 if elem > 0.5 else 0 for elem in y_predict]

		return np.array(y_predict_labels)[:, np.newaxis]