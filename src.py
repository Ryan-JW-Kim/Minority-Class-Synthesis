from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.hux import HUX
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import Hypervolume
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from sklearn.model_selection import StratifiedShuffleSplit

from joblib import Parallel, delayed

from scipy.stats import ranksums

import pickle
import os
import re

import pandas as pd
import numpy as np

def prepare_splits(x, y):
	train_split = StratifiedShuffleSplit(
		n_splits=31, 
		test_size=0.5
	)
	splits = []
	for train_idx, temp_idx in train_split.split(x, y):
		test_split = StratifiedShuffleSplit(
			n_splits=1, 
			test_size=0.5
		)
		test_idx, validation_idx = next(test_split.split(x[temp_idx], y[temp_idx]))

		validation_idx = temp_idx[validation_idx]
		test_idx = temp_idx[test_idx]
		
		splits.append((train_idx, validation_idx, test_idx))
	return splits

def create_model(variables, x_train, y_train, K):
	
	type_mappings = {}
	for variable_idx, variable_name in enumerate(variables['name']):
		variable_type = variables['type'][variable_idx]
		if variable_type not in type_mappings:
			type_mappings[variable_type] = []

		if variables['role'][variable_idx] == 'Feature':
			type_mappings[variable_type].append(variable_name)

	categorical_transformer = Pipeline(steps=[
		('imputer', SimpleImputer(strategy='most_frequent')),
		('onehot', OneHotEncoder(handle_unknown='ignore'))
	])
	numerical_transformer = Pipeline(steps=[
		('imputer', SimpleImputer(strategy='mean')),
		('scaler', StandardScaler())
	])

	numerical_features = []
	if 'Continuous' in type_mappings:
		for feature in type_mappings['Continuous']:
			numerical_features.append(feature)
	if 'Integer' in type_mappings:
		for feature in type_mappings['Integer']:
			numerical_features.append(feature)
			
	transformer_steps = []
	if numerical_features != []:
		transformer_steps.append(
			('num', numerical_transformer, numerical_features)
		)
	if 'Categorical' in type_mappings:
		transformer_steps.append(
			('cat', categorical_transformer, type_mappings['Categorical'])
		)
	preprocessor = ColumnTransformer(
		transformers=transformer_steps
	)
	model = Pipeline(steps=[
		('preprocessor', preprocessor),
		('classifier', KNeighborsClassifier(n_neighbors=K))
	])
	model.fit(x_train, y_train)
	
	return model

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
	
def load_datasets():
	from ucimlrepo import fetch_ucirepo 
	datasets = [
		(fetch_ucirepo(id=52), "ionosphere"),
		(fetch_ucirepo(id=43), "haberman"),
		(fetch_ucirepo(id=53), "iris0"),
		(fetch_ucirepo(id=42), "glass1"),
		(fetch_ucirepo(id=143), "australia"),
		(fetch_ucirepo(id=277), "thoracic"),
		(fetch_ucirepo(id=50), "segment0"),
		# (fetch_ucirepo(id=149), "vehicle0"),
		(fetch_ucirepo(id=109), "wine"),
		# (fetch_ucirepo(id=39), "ecoli"),
		(fetch_ucirepo(id=225), "ILPD"),
		(fetch_ucirepo(id=45), "heart_disease"),
		(fetch_ucirepo(id=17), "wisconsin"),
		# (fetch_ucirepo(id=73), "mushroom"),
		(fetch_ucirepo(id=94), "spambase"),
		(fetch_ucirepo(id=161), "mammographic"),
		# (fetch_ucirepo(id=12), "balance"),
		(fetch_ucirepo(id=110), "yeast1"),
		(fetch_ucirepo(id=451), "coimbra"),
		(fetch_ucirepo(id=244), "fertility")
	]

	return datasets