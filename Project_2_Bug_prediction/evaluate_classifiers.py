import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import wilcoxon
import json
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_recall_fscore_support, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt


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

print(paths_csv)

dataset = pd.read_csv(paths_csv[0])

X = dataset[[col for col in dataset.columns if col not in ['buggy', 'class']]]
y = dataset['buggy'].astype('int')

with open('best_parameters_for_models.json', 'r') as file:
    best_parameters = json.load(file)

decision_tree_params = best_parameters.get('Decision Tree', {})
naive_bayes_params = best_parameters.get('Gaussian Naive Bayes', {})
svm_params = best_parameters.get('Support Vector Machine', {})
mlp_params = best_parameters.get('Multilayer Perceptron', {})
random_forest_params = best_parameters.get('Random Forest', {})

models = {
    "Decision Tree": DecisionTreeClassifier(**decision_tree_params),
    "Gaussian Naive Bayes": GaussianNB(**naive_bayes_params),
    "Support Vector Machine": SVC(**svm_params),
    "Multilayer Perceptron": MLPClassifier(**mlp_params),
    "Random Forest": RandomForestClassifier(**random_forest_params)
}

n_splits = 5  # Number of folds
n_repeats = 20  # Number of repetitions
evaluations = []

# Evaluate biased classifier
biased_evaluations = []

kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    y_pred = [1] * len(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    biased_evaluations.append(report)

# Evaluate other classifiers
for model_name, model in models.items():
    print(f"Running cross-validation for {model_name}...")
    kf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    model_evaluations = []
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        model_evaluations.append(report)

    evaluations.append((model_name, model_evaluations))

# Instead of just f-scores, collect precision, recall, and f-scores for each model
scores = {
    model_name: {'precision': [], 'recall': [], 'f1': []}
    for model_name in models.keys()
}
scores['Biased'] = {'precision': [], 'recall': [], 'f1': []}

# Collect scores for biased classifier
for report in biased_evaluations:
    scores['Biased']['precision'].append(report['weighted avg']['precision'])
    scores['Biased']['recall'].append(report['weighted avg']['recall'])
    scores['Biased']['f1'].append(report['weighted avg']['f1-score'])

# Collect scores for other classifiers
for model_name, model_evaluations in evaluations:
    for report in model_evaluations:
        scores[model_name]['precision'].append(report['weighted avg']['precision'])
        scores[model_name]['recall'].append(report['weighted avg']['recall'])
        scores[model_name]['f1'].append(report['weighted avg']['f1-score'])

    for score_type in ['precision', 'recall', 'f1']:
        if score_type == 'f1':
            _, p_value = wilcoxon(scores['Biased'][score_type], scores[model_name][score_type])
            print(f"Wilcoxon test for {model_name} {score_type} = {p_value}")
        else:
            pass

# Plot box plots for precision, recall, and f-scores
for score_type in ['precision', 'recall', 'f1']:
    data = {model_name: model_scores[score_type] for model_name, model_scores in scores.items()}
    df = pd.DataFrame(data)
    df.boxplot()
    plt.title(f'Box Plot of {score_type.capitalize()}')
    plt.xlabel('Models')
    plt.ylabel(score_type)
    plt.show()