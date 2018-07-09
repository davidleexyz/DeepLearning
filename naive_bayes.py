import numpy as np 

class NaiveBayes():
	def __init__(self):
		self.classes = None
		self.class_prior_prob = []
		self.condition_prob = {}

	def calc_feature_prob(self, feature):
		feature_values = np.unique(feature)
		sample_count = len(feature)

		prob = {}

		for v in feature_values:
			prob[v] = np.sum(np.equal(feature, v)) / sample_count

		return prob

	def fit(self, X, y):
		self.classes = np.unique(y)
		class_num = len(self.classes)
		n_samples = len(y)
		n_features = np.array(X).shape[1]

		for c in self.classes:
			count = np.sum(np.equal(y, c))
			c_prob = count / n_samples
			self.class_prior_prob.append(c_prob)

		for c in self.classes:
			self.condition_prob[c] = {}
			for i in range(n_features):
				feature = X[np.equal(y, c)][:,i]
				self.condition_prob[c][i] = self.calc_feature_prob(feature)

		return

	def predict(self, x):
		label = -1
		best_feature_prob = 0

		for c in range(len(self.classes)):
			class_prob = self.class_prior_prob[c]
			feature_prob = self.condition_prob[self.classes[c]]
			current_feature_prob = 1.0
			j = 0
			for i in feature_prob.keys():
				current_feature_prob *= feature_prob[i][x[j]]
				j += 1

			if current_feature_prob*class_prob > best_feature_prob:
				best_feature_prob = current_feature_prob * class_prob
				label = self.classes[c]

		return label
