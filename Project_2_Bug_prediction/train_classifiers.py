import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import precision_recall_fscore_support

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from itertools import product
import os
import json

directory = '.'
paths_csv = []
substring_ratget = "_target"
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f) and f.endswith('.csv'):
        if substring_ratget in f:
            paths_csv.append(f)
        else:
            pass

# print(paths_csv)

dataset = pd.read_csv(paths_csv[0])

X = dataset[[col for col in dataset.columns if col not in ['buggy', 'class']]]
y = dataset['buggy'].astype('int')


models = {
    "Decision Tree": DecisionTreeClassifier,
    "Gaussian Naive Bayes": GaussianNB,
    "Support Vector Machine": SVC,
    "Multilayer Perceptron": MLPClassifier,
    "Random Forest": RandomForestClassifier
}

hyperparameters = {
    "Decision Tree": {
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"]
    },
    "Gaussian Naive Bayes": {},
    "Support Vector Machine": {
        "C": [1.0, 10.0],
        "kernel": ["linear", "rbf"]
    },
    "Multilayer Perceptron": {
        "hidden_layer_sizes": [(100,), (50, 50), (70, 30)],
        "activation": ["relu", "tanh", "logistic"],
        "solver" : ["sgd", "adam"],
        "max_iter": [500]
    },
    "Random Forest": {
        "n_estimators": [100, 200, 250],
        "max_depth": [None, 10, 15]
    }
}

best_scores = {}
best_params = {}
best_precisions = {}
best_recalls = {}
best_fscores = {}
n_folds = 5
for model_name, model_class in models.items():
    print(f'Now it goes {model_name}...')
    model_best_score = 0
    model_best_params = {}
    model_best_precision = 0
    model_best_recall = 0
    model_best_fscore = 0

    model_hyperparameters = hyperparameters[model_name]

    hyperparameter_combinations = product(*model_hyperparameters.values())

    for hyperparameter_values in hyperparameter_combinations:

        model = model_class(**dict(zip(model_hyperparameters.keys(), hyperparameter_values)))

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        model.fit(X_train, y_train)

        kf = KFold(n_splits=n_folds)
        scores = cross_val_score(model, X_val, y_val, cv=kf, scoring='f1')
        average_score = scores.mean()

        y_pred = model.predict(X_val)
        precision, recall, f_score, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')


        if average_score > model_best_score:
            model_best_score = average_score
            model_best_params = dict(zip(model_hyperparameters.keys(), hyperparameter_values))
            model_best_precision = precision
            model_best_recall = recall
            model_best_fscore = f_score


    best_scores[model_name] = model_best_score
    best_params[model_name] = model_best_params
    best_precisions[model_name] = model_best_precision
    best_recalls[model_name] = model_best_recall
    best_fscores[model_name] = model_best_fscore

print('--------------------------------------------------')
best_parameters_to_save = {}
for model_name, best_score in best_scores.items():
    best_parameters_to_save[model_name] = best_params[model_name]

    print("Best Accuracy score for", model_name, ":", best_score)
    print("Best precision for", model_name, ":", best_precisions[model_name])
    print("Best recall for", model_name, ":", best_recalls[model_name])
    print("Best fscore for", model_name, ":", best_fscores[model_name])
    print("Best parameters:", model_name, ":",  best_params[model_name])
    print()

with open('./best_parameters_for_models.json', 'w') as file:
    json.dump(best_parameters_to_save, file)

