from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.problem import Problem

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

from joblib import Parallel, delayed

import pandas as pd
import numpy as np

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

class GenericOptimizer(Problem):
	population_size = 100
	n_neighbours = 5
	sequential = False
	def __init__(self, X_train, y_train, X_val, y_val, objectives, exec_mode):
		
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
	def calculate_class1_precision(cls, x_train, y_train, x_val, y_val, n):
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
	def calculate_class1_recall(cls, x_train, y_train, x_val, y_val, n):
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
	