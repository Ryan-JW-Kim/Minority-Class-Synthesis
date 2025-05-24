from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, balanced_accuracy_score, roc_auc_score

from pymoo.operators.mutation.bitflip import BitflipMutation, Mutation
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.sampling.rnd import BinaryRandomSampling, Sampling
from pymoo.operators.crossover.hux import HUX
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import Hypervolume
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch

from scipy.stats import ranksums

from joblib import Parallel, delayed
from pathlib import Path
from io import StringIO
import pandas as pd
import numpy as np
import pickle
import os
import re

from main import *

import matplotlib.pyplot as plt

lists = []


	
with open('../data.pickle', 'rb') as fh:
	data_mapper = pickle.load(fh)

data_keys = list(data_mapper.keys())

results_by_name = {}
synthetic_by_name = {}
for file in os.listdir('results'):
	if 'csv' in file:
		synthetic_by_name[file] = pd.read_csv(f"results/{file}")
		
	else:
		with open(f"results/{file}", 'rb') as fh:
			results_by_name[file] = pickle.load(fh)
			
iter_by_datset_name = {}
for file in results_by_name:
	iter_name = file.replace(".result", '')
	iter_num = iter_name.split("_")[0]
	dataset_name = "_".join(iter_name.split("_")[1:])
	
	if dataset_name not in iter_by_datset_name:
		iter_by_datset_name[dataset_name] = []
	
	iter_by_datset_name[dataset_name].append(file)

for k in iter_by_datset_name:
	print(len(iter_by_datset_name[k]), repr(k))
	
for curr in iter_by_datset_name:
	validation_baseline = []
	test_baseline = []

	oversample_validation = []
	oversample_test = []

	record = []
	def execute_file(file):
		lists = {
		"Validation baseline auc": [],
		"Test baseline auc": [],

		"Validation baseline acc": [],
		"Test baseline acc": [],

		"Optimized Validation auc": [],
		"Optimized Test auc": [],
		"Ideal Test auc": [],

		"Optimized Validation acc": [],
		"Optimized Test acc": [],
		"Ideal Test acc": [],

		}
		result = results_by_name[file]
		synthetic_samples = synthetic_by_name[file.replace(".result", ".csv")]
		
		data_split = data_mapper[file.replace(".result", "")]
		x_train, y_train = data_split['x_train'], data_split['y_train']
		x_validation, y_validation = data_split['x_validation'], data_split['y_validation']
		x_test, y_test = data_split['x_test'], data_split['y_test']
		
		minority_label = pd.DataFrame(y_train).value_counts().argmin()


		# Fit with baseline train
		model = KNeighborsClassifier(n_neighbors=AUC_Optimizer.n_neighbours)
		model.fit(x_train, y_train)

		y_pred = model.predict(x_validation)
		baseline_validation_acc = accuracy_score(y_validation, y_pred)
		baseline_validation_auc = roc_auc_score(y_validation, y_pred)
		
		y_pred = model.predict(x_test)
		baseline_test_acc = accuracy_score(y_test, y_pred)
		baseline_test_auc = roc_auc_score(y_test, y_pred)

		x_SYNTH, y_SYNTH = np.concatenate((x_train, synthetic_samples)), np.concatenate((y_train, [minority_label] * len(synthetic_samples)))
	
		# Select ideal instance
		problem = AUC_Optimizer(
			x_SYNTH,
			y_SYNTH,
			x_validation,
			y_validation,
		)
		algorithm = NSGA2(
			pop_size=AUC_Optimizer.population_size, 
			sampling=DiverseCustomSampling(),
			crossover=HUX(), 
			mutation=BitflipMutation(), 
			eliminate_duplicates=True,
		)
		result = minimize(
			problem, 
			algorithm, 
			('n_gen', AUC_Optimizer.population_size), # <--- maybe increase
			save_history=False
		)
		
		validation_aucs = []
		test_aucs = []
		
		for instance in result.X:
			if np.sum(instance) >= AUC_Optimizer.n_neighbours:
				model = KNeighborsClassifier(n_neighbors=AUC_Optimizer.n_neighbours)
				model.fit(x_SYNTH[instance], y_SYNTH[instance])
				y_pred = model.predict(x_validation)
				validation_aucs.append(roc_auc_score(y_validation, y_pred))
				y_pred = model.predict(x_test)
				test_aucs.append(roc_auc_score(y_test, y_pred))
			else:
				validation_aucs.append(0)
				test_aucs.append(0)
				
		validation_idx = np.argmax(validation_aucs)
		test_idx = np.argmax(test_aucs)

		# Calculate metrics using ideal instance w.r.t validation AUC
		model = KNeighborsClassifier(n_neighbors=AUC_Optimizer.n_neighbours)
		model.fit(x_SYNTH[result.X[validation_idx]], y_SYNTH[result.X[validation_idx]])

		y_pred = model.predict(x_validation)
		optimized_validation_acc = accuracy_score(y_validation, y_pred)
		optimized_validation_auc = roc_auc_score(y_validation, y_pred)
		
		y_pred = model.predict(x_test)
		optimized_test_acc = accuracy_score(y_test, y_pred)
		optimized_test_auc = roc_auc_score(y_test, y_pred)

		# Calculate metrics using ideal instance w.r.t test AUC
		model = KNeighborsClassifier(n_neighbors=AUC_Optimizer.n_neighbours)
		model.fit(x_SYNTH[result.X[test_idx]], y_SYNTH[result.X[test_idx]])
		
		y_pred = model.predict(x_test)
		ideal_test_acc = accuracy_score(y_test, y_pred)
		ideal_test_auc = roc_auc_score(y_test, y_pred)

		lists["Validation baseline acc"].append(baseline_validation_acc)
		lists["Test baseline acc"].append(baseline_test_acc)

		lists["Validation baseline auc"].append(baseline_validation_auc)
		lists["Test baseline auc"].append(baseline_test_auc)

		lists["Optimized Validation auc"].append(optimized_validation_auc)
		lists["Optimized Test auc"].append(optimized_test_auc)
		lists["Ideal Test auc"].append(ideal_test_auc)

		lists["Optimized Validation acc"].append(optimized_validation_acc)
		lists["Optimized Test acc"].append(optimized_test_acc)
		lists["Ideal Test acc"].append(ideal_test_acc)
		return lists
	calcs = Parallel(n_jobs=-1)(delayed(execute_file)(file) for file in iter_by_datset_name[curr])

	lists = {
		"Validation baseline auc": [],
		"Test baseline auc": [],

		"Validation baseline acc": [],
		"Test baseline acc": [],

		"Optimized Validation auc": [],
		"Optimized Test auc": [],
		"Ideal Test auc": [],

		"Optimized Validation acc": [],
		"Optimized Test acc": [],
		"Ideal Test acc": [],
	}
	for ls in calcs:
		for key in ls:
			lists[key].append(ls[key])
			
	# counts = pd.DataFrame(y_train).value_counts()
	print(curr)

	print(f"Mean optimized validation acc diff {np.mean(np.subtract(lists['Optimized Validation acc'], lists['Validation baseline acc']))}")
	print(f"Mean optimized test acc diff       {np.mean(np.subtract(lists['Optimized Test acc'], lists['Test baseline acc']))}")
	print(f"Mean ideal test acc diff           {np.mean(np.subtract(lists['Ideal Test acc'], lists['Test baseline acc']))}")
	
	print(f"Mean optimized validation auc diff {np.mean(np.subtract(lists['Optimized Validation auc'], lists['Validation baseline auc']))}")
	print(f"Mean optimized test auc diff       {np.mean(np.subtract(lists['Optimized Test auc'], lists['Test baseline auc']))}")
	print(f"Mean ideal test auc diff           {np.mean(np.subtract(lists['Ideal Test auc'], lists['Test baseline auc']))}")
	
	print(f"\nValidation acc diff pval         {True if ranksums(lists['Validation baseline acc'], lists['Optimized Validation acc']).pvalue < 0.05 else False}")
	print(f"Test acc diff pval                 {True if ranksums(lists['Test baseline acc'], lists['Optimized Test acc']).pvalue < 0.05 else False}")
	print(f"Ideal Test acc diff pval           {True if ranksums(lists['Test baseline acc'], lists['Ideal Test acc']).pvalue < 0.05 else False}")

	print(f"\nValidation auc diff pval         {True if ranksums(lists['Validation baseline auc'], lists['Optimized Validation auc']).pvalue < 0.05 else False}")
	print(f"Test auc diff pval                 {True if ranksums(lists['Test baseline auc'], lists['Test baseline auc']).pvalue < 0.05 else False}")
	print(f"Ideal Test auc diff pval           {True if ranksums(lists['Test baseline auc'], lists['Ideal Test auc']).pvalue < 0.05 else False}")
	print("\n--\n")
	


	