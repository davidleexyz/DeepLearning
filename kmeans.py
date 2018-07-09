import numpy as np 
import random

class KMeans():
	def __init__(self, n_clusters):
		self.k = n_clusters

	def fit(self, data):
		self.centers = np.array(random.sample(list(data), self.k))

		self.initial_centers = np.copy(self.centers)

		old_assigns = None
		n_iters = 0

		while True:
			new_assigns = [self.classify(datapoint) for datapoint in data]

			if new_assigns == old_assigns:
				print(f"Training finished after {n_iters} interations!")
				return

			old_assigns = new_assigns
			n_iters += 1

			for id_ in range(self.k):
				points_idx = np.where(np.array(new_assigns) == id_)
				datapoints = data[points_idx]
				self.centers[id_] = datapoints.mean(axis=0)

	def l2_distance(self, datapoint):
		dists = np.sqrt(np.sum((self.centers - datapoint) ** 2, axis=1))
		return dists

	def classify(self, datapoint):
		dists = self.l2_distance(datapoint)
		return np.argmin(dists)