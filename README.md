# CP493-Research-Project

Using NSGA-II to explore novel method of improvment for multi-objective optimization on imbalanced datasets.

## Experiment outcome 1

In this experiment I tried multiple combinations of my varying execution configurations. For example, to try and increase the representativeness of the minority class I utilize an 'over sampling' scheme which duplicates all minority class samples of the training set and concatenates them onto training set itself. I also a attempt a biased initial population instantiation which increases the likely hood of genes being turned on (indicating inclusion of that sample) of which correspond to minority class samples. Then I utilize different sets of objectives and examine their effectiveness during optimization. The first of the objectives was a standard error plus number of examples minimization. Then, class 0 error and class 1 error, in which the fitness value of each individual is calculated for each class (minority and majority) seperately. Then, error plus number of examples, plus inverse f1 score.

### Outcome:

Oversampling works during optimization but overfits the validation set.
Biased initialization did not work very well, Azam predicts this is because of poor diversity in the initial population.

Best optimizer was the class sensitive error.
