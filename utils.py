from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

def split_and_scale_datasets(X, y, split_1=0.5, split_2=0.5, random_state=None, scale=True):
	X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=split_1, random_state=random_state, stratify=y)
	X_val, X_test, y_val, y_test =  train_test_split(X_temp, y_temp, test_size=split_2, random_state=random_state, stratify=y_temp)

	if scale:
		scaler = StandardScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)
		X_val = scaler.transform(X_val)

	return X_train, X_val, X_test, y_train, y_val, y_test

def set_summary(dataset, set_name, print_res=True):

	if "x" in set_name.lower():
		num_examples = dataset.shape[0]
		num_features = dataset.shape[1]
		
		if print_res:
			print(f"\nSummary for set {set_name}")
			print(f"\t- Num examples: {num_examples}")
			print(f"\t- Num features: {num_features}")

		else:
			return num_examples, num_features
		
	if "y" in set_name:
		counts = pd.DataFrame(dataset).value_counts()
		class_0_count = counts[0]
		class_1_count = counts[1]

		majority_class = max(class_0_count, class_1_count)
		minority_class = min(class_0_count, class_1_count)

		majority_class_name = 0 if majority_class == class_0_count else class_1_count

		if print_res:
			print(f"\nSummary for set {set_name}")
			print(f"\t- Number of examples in class 0: {class_0_count}")
			print(f"\t- Number of examples in class 1: {class_1_count}")
			print(f"\t- Total number of examples: {class_0_count + class_1_count}")
			print(f"\t- Imbalance ratio: {round(majority_class/minority_class, 4)} (Majority class is {majority_class_name})")		
		
		else:
			return class_0_count, class_1_count, round(majority_class/minority_class, 4)

def generate_PCA(X, y, plot_title):
	pca = PCA(n_components=2)
	X_pca = pca.fit_transform(X)
	plt.figure(figsize=(8, 6))
	plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k')
	plt.title(plot_title)
	plt.xlabel('Principal Component 1')
	plt.ylabel('Principal Component 2')
	return X_pca, pca

def parse_dataset(path, name, over_sample=False):
	global RANDOM_SEED
	try:
		df = pd.read_csv(path, delimiter=', ', engine='python')
		X = df.drop(columns='Class')
		y = df['Class']
	except KeyError:
		df = pd.read_csv(path, delimiter=',')
		X = df.drop(columns='Class')
		y = df['Class']

	# Generate train, validation, and test sets
	label_encoder = LabelEncoder()
	y_encoded = label_encoder.fit_transform(y)
	X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale_datasets(X, y_encoded)

	if over_sample:
		class_0_count, class_1_count, IR = set_summary(y_train, "y_train", False)
		if class_0_count > class_1_count:
			minority_class_indicies = np.where(y_train == 1)
		else:
			minority_class_indicies = np.where(y_train == 0)

		X_train = np.concatenate((X_train, X_train[minority_class_indicies]), axis=0)
		y_train = np.concatenate((y_train, y_train[minority_class_indicies]), axis=0)

	return [X, y_encoded, X_train, X_val, X_test, y_train, y_val, y_test, name]

def class_based_accuracy(model, x, y):
	class_1_indices = np.where(y==1)
	class_0_indices = np.where(y==0)

	class_1_x = x[class_1_indices]
	class_0_x = x[class_0_indices]

	class_1_y = y[class_1_indices]
	class_0_y = y[class_0_indices]
	
	class_1_pred = model.predict(class_1_x)
	class_1_acc = accuracy_score(class_1_y, class_1_pred)

	class_0_pred = model.predict(class_0_x)
	class_0_acc = accuracy_score(class_0_y, class_0_pred)

	overall_prediction = model.predict(x)
	overall_accuracy = accuracy_score(y, overall_prediction)

	return class_0_acc, class_1_acc, overall_accuracy

def assess_baseline_metrics(X_train, y_train, X_test, y_test):
	global N_NEIGHBOURS

	counts = pd.DataFrame(y_train).value_counts()

	# Determine baseline accuracy of classifier on all examples
	baseline_knn = KNeighborsClassifier(n_neighbors=N_NEIGHBOURS)
	baseline_knn.fit(X_train, y_train)
	class_0_baseline_testAcc, class_1_baseline_testAcc, baseline_testAcc = class_based_accuracy(baseline_knn, X_test, y_test)

	return counts, class_0_baseline_testAcc, class_1_baseline_testAcc, baseline_testAcc

def select_optimal_instance(X_train, y_train, X_val, y_val, result, N_NEIGHBOURS=5):

	fronts = NonDominatedSorting().do(result.F, only_non_dominated_front=True)
	_, pareto_indicies = np.unique(result.F[fronts], axis=0, return_index=True)

	best_instance_idx = 0
	best_acc = 0
	best_instance = None
	for idx, instance in enumerate(result.X[pareto_indicies]):
		x_filtered, y_filtered = X_train[instance], y_train[instance]
		if x_filtered.shape[0] < N_NEIGHBOURS: 
			acc = 1
		else:
			knn = KNeighborsClassifier(n_neighbors=N_NEIGHBOURS)
			knn.fit(x_filtered, y_filtered)
			y_pred = knn.predict(X_val)
			acc = accuracy_score(y_val, y_pred)
		
			if acc > best_acc:
				best_acc = acc
				best_instance_idx = idx
				best_instance = instance
			
	return best_acc, best_instance_idx, X_train[best_instance], y_train[best_instance]