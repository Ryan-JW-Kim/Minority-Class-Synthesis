from main import *

def overSample_randPop_execute(package):
	x, y, train_idx, validation_idx, test_idx, objectives, run_name = package
	x_train, y_train = x[train_idx], y[train_idx]
	x_validation, y_validation = x[validation_idx], y[validation_idx]
	x_train, y_train = over_sample(
		x_train, 
		y_train
	)
	problem = GenericOptimizer(
		x_train, 
		y_train, 
		x_validation, 
		y_validation,
		objectives,
		"Sequential"
	)	# BiasedBinarySampling(y_train, 0.5, 0.7)
	algorithm = NSGA2(
		pop_size=GenericOptimizer.population_size, 
		sampling=BinaryRandomSampling(), 
		crossover=HUX(), 
		mutation=BitflipMutation(), 
		eliminate_duplicates=True,
	)
	result = minimize(
		problem, 
		algorithm, 
		('n_gen', GenericOptimizer.population_size),
		save_history=True
	)
	package = {
		"name": run_name,
		"train": train_idx,
		"validation": validation_idx,
		"test": test_idx,
		"result": result
	}
	return package

def regularSample_randPop_execute(package):
	x, y, train_idx, validation_idx, test_idx, objectives, run_name = package
	x_train, y_train = x[train_idx], y[train_idx]
	x_validation, y_validation = x[validation_idx], y[validation_idx]
	problem = GenericOptimizer(
		x_train, 
		y_train, 
		x_validation, 
		y_validation,
		objectives,
		"Sequential"
	)	# BiasedBinarySampling(y_train, 0.5, 0.7)
	algorithm = NSGA2(
		pop_size=GenericOptimizer.population_size, 
		sampling=BinaryRandomSampling(), 
		crossover=HUX(), 
		mutation=BitflipMutation(), 
		eliminate_duplicates=True,
	)
	result = minimize(
		problem, 
		algorithm, 
		('n_gen', GenericOptimizer.population_size),
		save_history=True
	)
	package = {
		"name": run_name,
		"train": train_idx,
		"validation": validation_idx,
		"test": test_idx,
		"result": result
	}
	return package

def overSample_biasPop_execute(package):
	x, y, train_idx, validation_idx, test_idx, objectives, run_name = package
	x_train, y_train = x[train_idx], y[train_idx]
	x_validation, y_validation = x[validation_idx], y[validation_idx]
	x_train, y_train = over_sample(
		x_train, 
		y_train
	)
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
		sampling=BiasedBinarySampling(y_train, 0.5, 0.7), 
		crossover=HUX(), 
		mutation=BitflipMutation(), 
		eliminate_duplicates=True,
	)
	result = minimize(
		problem, 
		algorithm, 
		('n_gen', GenericOptimizer.population_size),
		save_history=True
	)
	package = {
		"name": run_name,
		"train": train_idx,
		"validation": validation_idx,
		"test": test_idx,
		"result": result
	}
	return package

def regularSample_biasPop_execute(package):
	x, y, train_idx, validation_idx, test_idx, objectives, run_name = package
	x_train, y_train = x[train_idx], y[train_idx]
	x_validation, y_validation = x[validation_idx], y[validation_idx]
	problem = GenericOptimizer(
		x_train, 
		y_train, 
		x_validation, 
		y_validation,
		objectives,
		"Sequential"
	)	# BiasedBinarySampling(y_train, 0.5, 0.7)
	algorithm = NSGA2(
		pop_size=GenericOptimizer.population_size, 
		sampling=BiasedBinarySampling(y_train, 0.5, 0.7), 
		crossover=HUX(), 
		mutation=BitflipMutation(), 
		eliminate_duplicates=True,
	)
	result = minimize(
		problem, 
		algorithm, 
		('n_gen', GenericOptimizer.population_size),
		save_history=True
	)
	package = {
		"name": run_name,
		"train": train_idx,
		"validation": validation_idx,
		"test": test_idx,
		"result": result
	}
	return package

def AutoEncoder_execute(package):
	x, y, train_idx, validation_idx, run_name = package
	x_train, y_train = x[train_idx], y[train_idx]
	x_validation, y_validation = x[validation_idx], y[validation_idx]

	mutation_operator = CustomMutation(
		x_train, y_train, 
		x_validation, y_validation,
		prediction_threshold=0.5
	)
	algorithm = NSGA2(
		pop_size=GenericOptimizer.population_size, 
		sampling=BinaryRandomSampling(), 
		crossover=HUX(), 
		mutation=mutation_operator, 
		eliminate_duplicates=True
	)
	problem = GenericOptimizer(
		x_train, 
		y_train, 
		x_validation, 
		y_validation,
		[CustomMutation.primary_objective, GenericOptimizer.calculate_num_examples],
		"Sequential"
	)
	result = minimize(
		problem, 
		algorithm, 
		('n_gen', GenericOptimizer.population_size),
		verbose=False,
		save_history=True
	)

	mutation_operator = None

	package = {
		"name": run_name,
		"train": train_idx,
		"validation": validation_idx,
		"test": test_idx,
		"result": result
	}
	return package