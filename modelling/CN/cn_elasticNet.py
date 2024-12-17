# import pandas as pd
# from sklearn.linear_model import ElasticNet
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
# from sklearn.impute import KNNImputer
#
# # Load data
# data = pd.read_csv('data/cn_preprocessed.csv')
#
# # Split data
# X = data.drop('last_DX', axis=1)
# y = data['last_DX']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Create pipeline
# pipe = Pipeline([
#     ('imputer', KNNImputer()),
#     ('scaler', StandardScaler()),
#     ('classifier', ElasticNet())
# ])
# #
# # # Grid search
# param_grid = {
#     'classifier__alpha': [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'classifier__l1_ratio': [0.1, 0.5, 0.7, 0.8, 0.9, 1]
# }
#
# grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc')
# grid.fit(X_train, y_train)
#
# # print best parameters
# print(grid.best_params_)
#
# # Evaluate model on train and test set
# y_train_pred = grid.predict_proba(X_train)
# y_test_pred = grid.predict_proba(X_test)
#
# train_auc = roc_auc_score(y_train, y_train_pred[:, 1])
# test_auc = roc_auc_score(y_test, y_test_pred[:, 1])
#
# print(f'Train AUC: {train_auc}')
# print(f'Test AUC: {test_auc}')

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# Load data
data = pd.read_csv('data/cn_preprocessed.csv')

# Define features and target
X = data.drop('last_DX', axis=1)
y = data['last_DX']

# create pipeline
pipe = Pipeline([
    ('imputer', KNNImputer()),
    ('scaler', StandardScaler()),
    ('classifier', ElasticNet())
])

# define hyperparameter grid
param_grid = {
    'classifier__alpha': [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'classifier__l1_ratio': [0.1, 0.5, 0.7, 0.8, 0.9, 1]
}

# Define number of folds based on data size
n_folds = len(X) // 3
remaining_rows = len(X) % 3

# Placeholder for predictions with indices
all_predictions = []

# Nested cross-validation procedure
used_indices = set()
# turn y numeric
y = y.replace({'CN': 0, 'MCI_AD': 1})
X.shape
for fold_idx in range(n_folds):
    print(f"Processing fold {fold_idx + 1}/{n_folds}")
    # Randomly select 3 rows for the test set
    test_indices = np.random.choice(list(X.index.difference(used_indices)), size=3, replace=False)
    used_indices.update(test_indices)
    train_indices = X.index.difference(test_indices)

    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]

    # Inner cross-validation for hyperparameter tuning
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_idx)
    grid_search = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Best model prediction on the test set
    best_model = grid_search.best_estimator_
    y_test_pred = best_model.predict_proba(X_test)[:, 1]

    # Store predictions with indices
    all_predictions.extend(zip(test_indices, y_test_pred))

# Handle remaining rows
if remaining_rows > 0:
    remaining_indices = list(X.index.difference(used_indices))
    X_train, X_test = X.loc[X.index.difference(remaining_indices)], X.loc[remaining_indices]
    y_train, y_test = y.loc[X.index.difference(remaining_indices)], y.loc[remaining_indices]

    # Inner cross-validation for hyperparameter tuning
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Best model prediction on the remaining test set
    best_model = grid_search.best_estimator_
    y_test_pred = best_model.predict_proba(X_test)[:, 1]

    # Store predictions with indices
    all_predictions.extend(zip(remaining_indices, y_test_pred))

# Sort predictions by index to match original y order
all_predictions_sorted = sorted(all_predictions, key=lambda x: x[0])
final_predictions = [pred for _, pred in all_predictions_sorted]

# Evaluate overall performance
auc_score = roc_auc_score(y, final_predictions)
# # calculate accuracy at the youden index
# youden_index = np.argmax(np.array(final_predictions)) # find the index of the maximum prediction
# youden_pred = np.zeros(len(final_predictions)) # initialize all predictions to 0
# youden_pred[youden_index] = 1
# accuracy = sum(y == youden_pred) / len(y)
# # calculate precision, recall, and f1 score at the youden index
# confusion_matrix = confusion_matrix(y, youden_pred)
# precision = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1])
# recall = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])
# f1_score = 2 * (precision * recall) / (precision + recall)
# print(f"Overall Accuracy: {accuracy}")
# print(f"Overall Precision: {precision}")
# print(f"Overall Recall: {recall}")
# print(f"Overall F1 Score: {f1_score}")
print(f"Overall AUC: {auc_score}")


