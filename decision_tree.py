import numpy as np 

class DecisionTreeID3():
	def __init__(self):
		self.dataset = None
		self.tree = None
	
	def calc_shannon_entropy(self, dataset):
		n_samples = np.array(dataset).shape[0]
		y = np.array(dataset)[:, -1]

		entropy = 0.0
		target_counts = {}
		for target in y:
			if target not in target_counts.keys():
				target_counts[target] = 0
			target_counts[target] += 1

		for key in target_counts.keys():
			prob = float(target_counts[key]) / n_samples
			entropy -= prob * np.log2(prob)

		return entropy

	def split_dataset(self, dataset, f_index, f_value):
		n_samples, n_features = np.array(dataset).shape
		feature = np.array(dataset)[:, f_index]
		x = dataset[:, [i for i in range(n_features) if i != f_index]]
		ret = []
		for i in range(len(feature)):
			if feature[i] == f_value:
				ret.append(i)

		return x[ret,:]

	def choose_best_feature_to_split_dataset(self, dataset):
		n_samples, n_features = np.array(dataset).shape

		dataset_entropy = self.calc_shannon_entropy(dataset)
		best_info_gain = 0.0
		best_feature_index = -1

		for i in range(n_features-1):
			feature = dataset[:,i]
			feature_values = set(feature)
			for value in feature_values:
				split_dataset = self.split_dataset(dataset, i, value)
				split_prob = split_dataset.shape[0] / n_samples
				split_entropy += prob * self.calc_shannon_entropy(split_dataset)

			info_gain = dataset - split_entropy
			if info_gain > best_info_gain:
				best_info_gain = info_gain
				best_feature_index = i
		return best_feature_index

	def max_target_count(self, y):
		label_count = {}
		for value in y:
			if value not in label_count.keys():
				label_count[value] = 0
			label_count += 1

		sorted_label_count = sorted(label_count.iteritems(), key=lambda x:x[1], reverse=True)
		return sorted_label_count[0][0]

	def create_tree(self, dataset, feature_indexs):
		n_samples, n_features = np.array(dataset).shape

		x = np.array(dataset)[:, :-1]
		y = np.array(dataset)[:, -1]

		if len(feature_indexs) == 0:
			return self.max_target_count(y)

		targets = list(y)
		if targets.count(targets[0]) == len(targets):
			return targets[0]

		best_feature_index = self.choose_best_feature_to_split_dataset(dataset)
		feature_indexs.remove(best_feature_index)
		tree = { best_feature_index : {}}
		best_feature_values = x[:, best_feature_index]
		values_set = set(best_feature_values)

		for value in values_set:
			split_dataset = self.split_dataset(dataset, best_feature_index, value)
			tree[best_feature_index][value] = self.create_tree(split_dataset, feature_indexs)

		return tree

	def fit(self, dataset):
		_, n_features = np.array(dataset).shape
		feature_indexs = range(n_features)
		self.dataset = dataset
		self.tree = self.create_tree(dataset, feature_indexs)
		return

	def predict(self, x):
		if self.tree == None:
			print("Didn't fit decision tree model")
			return

		return classify(self.tree, x)

	def classify(tree, sample):
		feature_index = tree.keys()
		feature_tree = tree[feature_index]

		sample_feature = sample[feature_index]
		sub_tree = feature_tree[sample_feature]
		if isinstance(sub_tree, dict):
			label = classify(sub_tree, sample)
		else:
			label = sub_tree

		return label