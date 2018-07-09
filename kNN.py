import numpy as np 

class kNN():

	def __init__(self):
		pass

	def fit(self, x, y):
		self.data = x
		self.targets = y

	def euclidean_distance(self, x):
		if x.ndim == 1:
			l2 = np.sqrt(np.sum((self.data - x) ** 2, axis = 1))

		if x.ndim == 2:
			n_samples, _ = x.shape
			l2 = [np.sqrt(np.sum((self.data - x[i]) ** 2, axis = 1)) for i in range(n_samples)]

		return np.array(l2)

	def predict(self, x, k=1):
		dists = self.euclidean_distance(x)

		if x.ndim == 1:
			if k == 1:
				nn = np.argmin(dists)
				return self.targets[nn]
			else:
				knn = np.argsort(dists)[:k]
				y_knn = self.targets[knn]
				max_vote = max(y_knn, key=list(y_knn).count)
				return max_vote

		if x.ndim == 2:
			knn = np.argsort(dists)[:, :k]
			y_knn = self.targets[knn]
			if k == 1:
				return y_knn.T 
			else:
				n_samples, _ = x.shape
				max_votes = [max(y_knn[i], key=list(y_knn[i]).count) for i in range(n_samples)]
				return max_votes
