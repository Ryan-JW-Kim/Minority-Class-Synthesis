from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.hux import HUX
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import Hypervolume
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from joblib import Parallel, delayed

from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch

from scipy.stats import ranksums
import matplotlib.pyplot as plt
import torch.nn as nn

import pandas as pd
import numpy as np

import dill

import pickle
import os
import re

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

class GenericOptimizer(Problem):
	population_size = 100
	n_neighbours = 5
	sequential = False
	def __init__(self, X_train, y_train, X_val, y_val, objectives, exec_mode):
		self.mutation_history = {}
		self.generation_number = 0

		self.exec_mode = exec_mode

		self.X_train = X_train
		self.y_train = y_train

		self.X_val = X_val
		self.y_val = y_val

		self.training_data = X_train
		self.n_instances = X_train.shape[0]
		
		self.objectives = objectives

		super().__init__(
			n_var=self.n_instances,
			n_obj=len(objectives),               
			n_constr=0,            
			xl=0,                  
			xu=1,                  
			type_var=np.bool_,     
		)

	def _evaluate(self, x, out, *args, **kwargs):
		
		if self.exec_mode == "sequential":
			metrics = []
			for objective in self.objectives:
				metrics.append(self.eval_objective((objective, x)))
		else:
			metrics = Parallel(n_jobs=-1)(delayed(self.eval_objective)((objective, x)) for objective in self.objectives)
		
		self.generation_number += 1

		out["F"] = np.column_stack(metrics)

	def eval_objective(self, pack):
		objective, x = pack
			
		if "calculate_num_examples" in repr(objective):
			return GenericOptimizer.calculate_num_examples(x)

		elif "calculate_IR" in repr(objective):
			vals = []
			for instance in x:
				vals.append(GenericOptimizer.calculate_IR(self.y_train[instance]))
			return vals
		
		else:
			vals = []
			for instance in x:
				vals.append(objective(
					self.X_train[instance],
					self.y_train[instance],
					self.X_val,
					self.y_val,
					GenericOptimizer.n_neighbours
				))
			return vals

	@classmethod
	def calculate_IR(cls, y):
		df = pd.DataFrame(y).value_counts()
		return (df[1]/df[0]) if df.min() == 0 else (df[0]/df[1])
	
	@classmethod
	def filter_by_class(cls, x, y, label):
		indices = np.where(y==label)
		return x[indices], y[indices]
	
	@classmethod
	def calculate_overall_error(cls, x_train, y_train, x_val, y_val, n):
				
		num_included_instances = x_train.shape[0]

		if num_included_instances >= n:
			optimization_knn = KNeighborsClassifier(n_neighbors=n)
			optimization_knn.fit(x_train, y_train)

			y_pred = optimization_knn.predict(x_val)
			acc = accuracy_score(y_val, y_pred)
			return 1-acc
		else:
			return 1

	@classmethod
	def calculate_class0_error(cls, x_train, y_train, x_val, y_val, n):
		class0_x_train, class0_y_train = cls.filter_by_class(x_train, y_train, 0)
		err = cls.calculate_overall_error(
			class0_x_train,
			class0_y_train,
			x_val,
			y_val,
			n
		)
		return err

	@classmethod
	def calculate_class1_error(cls, x_train, y_train, x_val, y_val, n):
		class0_x_train, class0_y_train = cls.filter_by_class(x_train, y_train, 1)
		err = cls.calculate_overall_error(
			class0_x_train,
			class0_y_train,
			x_val,
			y_val,
			n
		)
		return err

	@classmethod
	def calculate_overall_inverse_f1(cls, x_train, y_train, x_val, y_val, n):
				
		num_included_instances = x_train.shape[0]

		if num_included_instances >= n:
			optimization_knn = KNeighborsClassifier(n_neighbors=n)
			optimization_knn.fit(x_train, y_train)

			y_pred = optimization_knn.predict(x_val)
			f1 = f1_score(y_val, y_pred)
			return 1-f1
		else:
			return 1

	@classmethod
	def calculate_class0_inverse_f1(cls, x_train, y_train, x_val, y_val, n):
		class0_x_train, class0_y_train = cls.filter_by_class(x_train, y_train, 0)
		inv_f1 = cls.calculate_overall_inverse_f1(
			class0_x_train,
			class0_y_train,
			x_val,
			y_val,
			n
		)
		return inv_f1

	@classmethod
	def calculate_class1_inverse_f1(cls, x_train, y_train, x_val, y_val, n):
		class0_x_train, class0_y_train = cls.filter_by_class(x_train, y_train, 1)
		inv_f1 = cls.calculate_overall_inverse_f1(
			class0_x_train,
			class0_y_train,
			x_val,
			y_val,
			n
		)
		return inv_f1
	
	@classmethod
	def calculate_overall_inverse_precision(cls, x_train, y_train, x_val, y_val, n):
				
		num_included_instances = x_train.shape[0]

		if num_included_instances >= n:
			optimization_knn = KNeighborsClassifier(n_neighbors=n)
			optimization_knn.fit(x_train, y_train)

			y_pred = optimization_knn.predict(x_val)
			prec = precision_score(y_val, y_pred)
			return 1-prec
		else:
			return 1

	@classmethod
	def calculate_class0_inverse_precision(cls, x_train, y_train, x_val, y_val, n):
		class0_x_train, class0_y_train = cls.filter_by_class(x_train, y_train, 0)
		inv_prec = cls.calculate_overall_inverse_precision(
			class0_x_train,
			class0_y_train,
			x_val,
			y_val,
			n
		)
		return inv_prec

	@classmethod
	def calculate_class1_inverse_precision(cls, x_train, y_train, x_val, y_val, n):
		class0_x_train, class0_y_train = cls.filter_by_class(x_train, y_train, 1)
		inv_prec = cls.calculate_overall_inverse_precision(
			class0_x_train,
			class0_y_train,
			x_val,
			y_val,
			n
		)
		return inv_prec
		
	@classmethod
	def calculate_overall_inverse_recall(cls, x_train, y_train, x_val, y_val, n):
				
		num_included_instances = x_train.shape[0]

		if num_included_instances >= n:
			optimization_knn = KNeighborsClassifier(n_neighbors=n)
			optimization_knn.fit(x_train, y_train)

			y_pred = optimization_knn.predict(x_val)
			recall = recall_score(y_val, y_pred)
			return 1-recall
		else:
			return 1

	@classmethod
	def calculate_class0_inverse_recall(cls, x_train, y_train, x_val, y_val, n):
		class0_x_train, class0_y_train = cls.filter_by_class(x_train, y_train, 0)
		inv_recall = cls.calculate_overall_inverse_recall(
			class0_x_train,
			class0_y_train,
			x_val,
			y_val,
			n
		)
		return inv_recall

	@classmethod
	def calculate_class1_inverse_recall(cls, x_train, y_train, x_val, y_val, n):
		class0_x_train, class0_y_train = cls.filter_by_class(x_train, y_train, 1)
		inv_recall = cls.calculate_overall_inverse_recall(
			class0_x_train,
			class0_y_train,
			x_val,
			y_val,
			n
		)
		return inv_recall
	
	@classmethod
	def calculate_num_examples(cls, instances):
		return np.sum(instances, axis=1)

	@classmethod
	def quantify_performance(cls, population, objectives, x_train, y_train, x_validation, y_validation, x_test, y_test):
		pass

	@classmethod
	def unbound_eval_objectives(cls, objective, instances, x_train, y_train, x_validation, y_validation):
		if "calculate_num_examples" in repr(objective):
			return GenericOptimizer.calculate_num_examples(instances)

		elif "calculate_IR" in repr(objective):
			vals = []
			for instance in instances:
				vals.append(GenericOptimizer.calculate_IR(y_train[instance]))
			return vals
		
		else:
			vals = []
			for instance in instances:
				vals.append(objective(
					x_train[instance],
					y_train[instance],
					x_validation,
					y_validation,
					GenericOptimizer.n_neighbours
				))
			return vals
		
	@classmethod
	def calculate_optimal_instance(cls, x_train, y_train, x_val, y_val, metrics, population, n):

		fronts = NonDominatedSorting().do(metrics, only_non_dominated_front=True)
		_, pareto_indicies = np.unique(metrics[fronts], axis=0, return_index=True)

		best_acc = 0
		best_instance = None
		for idx, instance in enumerate(population[pareto_indicies]):
			x_filtered, y_filtered = x_train[instance], y_train[instance]
			if x_filtered.shape[0] < n: 
				acc = 0
			else:
				knn = KNeighborsClassifier(n_neighbors=n)
				knn.fit(x_filtered, y_filtered)
				y_pred = knn.predict(x_val)
				acc = accuracy_score(y_val, y_pred)
			
				if acc > best_acc:
					best_acc = acc
					best_instance = instance
				
		return pareto_indicies, x_train[best_instance], y_train[best_instance]
	  
def prepare_splits(path):
	try:
		df = pd.read_csv(path, delimiter=', ', engine='python')
		x = df.drop(columns='Class')
		y = df['Class']
	except KeyError:
		df = pd.read_csv(path, delimiter=',')
		x = df.drop(columns='Class')
		y = df['Class']

	x = np.array(x)
	
	label_encoder = LabelEncoder()
	y = label_encoder.fit_transform(y)
	
	train_split = StratifiedShuffleSplit(
		n_splits=31,
		test_size=0.5,
	)

	splits = []

	for train_idx, temp_idx in train_split.split(x, y):

		test_split = StratifiedShuffleSplit(
			n_splits=1,
			test_size=0.5
		)

		x_temp, y_temp = x[temp_idx], y[temp_idx]

		test_idx, validation_idx = next(test_split.split(x_temp, y_temp))

		validation_idx = temp_idx[validation_idx]
		test_idx = temp_idx[test_idx]

		splits.append((train_idx, validation_idx, test_idx))
	
	return x, y, splits

def over_sample(x, y):
	
	counts = pd.DataFrame(y).value_counts()

	if counts[0] < counts[1]:
		minority_class_indicies = np.where(y == 1)
	else:
		minority_class_indicies = np.where(y == 0)

	over_sampled_x = np.concatenate((x, x[minority_class_indicies]), axis=0)
	over_sampled_y = np.concatenate((y, y[minority_class_indicies]), axis=0)

	return over_sampled_x, over_sampled_y

class CustomDataset(Dataset):
	def __init__(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train
	def __len__(self):
		return self.x_train.shape[0]
	def __getitem__(self, ind):
		x = self.x_train[ind]
		y = self.y_train[ind]
		return x, y

class MLP(nn.Module):
	def __init__(self, input_dim):
		super(MLP, self).__init__()
		self.linear1 = nn.Linear(input_dim, input_dim//2)
		self.relu1 = nn.ReLU()
		self.linear2 = nn.Linear(input_dim//2, input_dim//3)
		self.relu2 = nn.ReLU()
		self.linear3 = nn.Linear(input_dim//3, input_dim)

	def forward(self, x):
		x = self.linear1(x)
		x = self.relu1(x)
		x = self.linear2(x)
		x = self.relu2(x)
		x = self.linear3(x)
		return x
	
class CustomMutation(Mutation):
	curr_MLP = None
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	num_synthetic_examples = 500000
	train_epochs = 50
	batch_size = 215
	primary_objective = GenericOptimizer.calculate_overall_error
	secondary_objectives = [
		[GenericOptimizer.calculate_num_examples],
		[GenericOptimizer.calculate_class0_error],
		[GenericOptimizer.calculate_class0_inverse_f1],
		[GenericOptimizer.calculate_class0_inverse_precision],
		[GenericOptimizer.calculate_class0_inverse_recall],
		[GenericOptimizer.calculate_class1_error],
		[GenericOptimizer.calculate_class1_inverse_f1],
		[GenericOptimizer.calculate_class1_inverse_precision],
		[GenericOptimizer.calculate_class1_inverse_recall],
		[GenericOptimizer.calculate_overall_inverse_f1],
		[GenericOptimizer.calculate_overall_inverse_precision],
		[GenericOptimizer.calculate_overall_inverse_recall],
		[GenericOptimizer.calculate_class0_inverse_precision, GenericOptimizer.calculate_class1_inverse_precision],
		[GenericOptimizer.calculate_class0_inverse_recall, GenericOptimizer.calculate_class1_inverse_recall],
		[GenericOptimizer.calculate_class0_inverse_f1, GenericOptimizer.calculate_class1_inverse_f1],
		[GenericOptimizer.calculate_class0_error, GenericOptimizer.calculate_class1_error],
	]

	def __init__(self, x_train, y_train, x_validation, y_validation, prediction_threshold=0.5):
		super().__init__()
		self.prediction_thresh = prediction_threshold
		synthesized_x, synthesized_y = CustomMutation.create_training_data(x_train, y_train, x_validation, y_validation)
		self.model = CustomMutation.train_mutation(synthesized_x, synthesized_y)

	def _do(self, problem, X, **kwargs):

		int_x = np.array(X, dtype=np.float32)
		dataset = CustomDataset(int_x, int_x)
		loader = DataLoader(dataset, batch_size=X.shape[0], shuffle=False)

		self.model.eval()
		with torch.no_grad():
			for data, _ in loader:
				data = data.to(CustomMutation.device)
				outputs = self.model(data)
				predictions = (outputs > self.prediction_thresh).bool()

		prediction = np.array(predictions)

		total_number_of_genes = X.shape[0] * X.shape[1]
		genes_effected = np.sum(X ^ prediction)

		if problem.generation_number not in problem.mutation_history:
			problem.mutation_history[problem.generation_number] = []
		
		problem.mutation_history[problem.generation_number].append(genes_effected/total_number_of_genes)
		return prediction

	@classmethod
	def train_mutation(cls, x_train, y_train):
		train_set = CustomDataset(x_train, y_train)
		input_dim = x_train.shape[1]
		batch_size = cls.batch_size
		train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		 
		model = MLP(input_dim).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
		criterion = nn.BCEWithLogitsLoss()

		model.train()
		for epoch in range(cls.train_epochs):
			losses = []
			for batch_num, input_data in enumerate(train_loader):
				optimizer.zero_grad()
				x, y = input_data
				x, y = x.to(device).float(), y.to(device)

				output = model(x)
				loss = criterion(output, y)
				loss.backward()
				losses.append(loss.item())
				optimizer.step()

		return model
	
	@classmethod
	def create_training_data(cls, x_train, y_train, x_validation, y_validation):
		
		synthesizing_splits = StratifiedShuffleSplit(
			n_splits=len(cls.secondary_objectives), # create a split for each secondary objective
			test_size=0.5, # Half the validation set is randomly excluded
		)	
		packages = []
		for idx, (sub_validation_idx, _) in enumerate(synthesizing_splits.split(x_validation, y_validation)):
			packages.append((
				cls.secondary_objectives[idx],
				x_train,
				y_train,
				x_validation[sub_validation_idx],
				y_validation[sub_validation_idx]
			))

		# Execute optimization and extract the final populations
		populations = Parallel(n_jobs=-1)(delayed(cls.execute_training_data_gen)(package) for package in packages)
		
		# Aggregate all populations into single list containing every unique instance
		all_instances = []
		for population in populations:
			for individual in population.pop:
				all_instances.append(list(individual.X))
				
		all_instances = np.array(all_instances)

		# Create synthetic examples by adding randin noise to each instance. Repeat until threshold is reached.
		synthetic_x, synthetic_y = [], []
		while len(synthetic_x) < cls.num_synthetic_examples:
			
			for y_true in all_instances:
				x_noised = []
				for idx, probability in enumerate(np.random.uniform(0.1, 1.0, y_true.shape[0])):
					if probability < 0.85:
						x_noised.append(y_true[idx])                
					else:
						x_noised.append(0 if y_true[idx] == 1 else 1)

				synthetic_x.append(np.array(x_noised, dtype=np.float32))
				synthetic_y.append(np.array(y_true, dtype=np.float32))

		return np.array(synthetic_x), np.array(synthetic_y)
	
	@classmethod
	def execute_training_data_gen(cls, package):

		objectives, x_train, y_train, x_validation, y_validation = package
		
		objectives.append(cls.primary_objective)

		problem = GenericOptimizer(
			x_train, 
			y_train, 
			x_validation, 
			y_validation,
			objectives,
			"Sequential"
		)

		algorithm = NSGA2(
			pop_size=GenericOptimizer.population_size, 
			sampling=BinaryRandomSampling(), 
			crossover=HUX(), 
			mutation=BitflipMutation(), 
			eliminate_duplicates=True
		)
		result = minimize(
			problem, 
			algorithm, 
			('n_gen', GenericOptimizer.population_size)
		)
		
		return result
	
class BitflipMutation(Mutation):

	def _do(self, problem, X, **kwargs):
		
		prob_var = self.get_prob_var(problem, size=(len(X), 1))
		Xp = np.copy(X)
		flip = np.random.random(X.shape) < prob_var
		Xp[flip] = ~X[flip]
		
		total_number_of_genes = X.shape[0] * X.shape[1]
		genes_effected = np.sum(X ^ Xp)

		if problem.generation_number not in problem.mutation_history:
			problem.mutation_history[problem.generation_number] = []
		
		problem.mutation_history[problem.generation_number].append(genes_effected/total_number_of_genes)

		return Xp

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
	
