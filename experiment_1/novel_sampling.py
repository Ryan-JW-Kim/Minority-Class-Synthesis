from pymoo.core.sampling import Sampling
import pandas as pd
import numpy as np

def over_sample(x, y):
	
	counts = pd.DataFrame(y).value_counts()

	if counts[0] < counts[1]:
		minority_class_indicies = np.where(y == 1)
	else:
		minority_class_indicies = np.where(y == 0)

	over_sampled_x = np.concatenate((x, x[minority_class_indicies]), axis=0)
	over_sampled_y = np.concatenate((y, y[minority_class_indicies]), axis=0)

	return over_sampled_x, over_sampled_y

class BiasedBinarySampling(Sampling):
	def __init__(self, labels, major_prob, minor_prob):
		
		self.labels = labels
		counts = pd.DataFrame(labels).value_counts()
		if counts[0] > counts[1]:
			self.c0_thresh = major_prob
			self.c1_thresh = minor_prob
		else:
			self.c0_thresh = minor_prob
			self.c1_thresh = major_prob

		super().__init__()

	def _do(self, problem, n_samples, **kwargs):

		rands = np.random.random((n_samples, problem.n_var))
		init_pops = np.zeros((n_samples, problem.n_var), dtype=bool)
		for idx, label in enumerate(self.labels):
			if label == 0:
				init_pops[:, idx] = (rands[:, idx] < self.c0_thresh).astype(bool)
			if label == 1:
				init_pops[:, idx] = (rands[:, idx] < self.c1_thresh).astype(bool)


		return init_pops

class InheritedSampling(Sampling):
	def __init__(self, pareto_front, mutation_prob, num_rows_inherited):
		
		self.parent = pareto_front
		self.thresh = mutation_prob
		self.inherit_thresh = num_rows_inherited
		super().__init__()

	def _do(self, problem, n_samples, **kwargs):

		init_pops = np.zeros((n_samples, problem.n_var), dtype=bool)
		rands = np.random.random((n_samples, problem.n_var))
		for i in range(init_pops.shape[0]):
			for j in range(init_pops.shape[1]):

				if i < self.inherit_thresh:
					if rands[i, j] < self.thresh:
						init_pops[i, j] = 0 if self.parent[i, j] == 1 else 1
					else:
						init_pops[i, j] = self.parent[i, j]
				else:
					init_pops[i, j] = 0 if rands[i, j] < 0.5 else 1

		return init_pops