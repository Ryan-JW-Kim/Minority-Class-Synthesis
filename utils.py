import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, OneHotEncoder


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
