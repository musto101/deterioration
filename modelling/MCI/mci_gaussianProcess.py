import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer

# Load data
data = pd.read_csv('data/mci_preprocessed.csv')

# Split data
X = data.drop('last_DX', axis=1)
y = data['last_DX']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipe = Pipeline([
    ('imputer', KNNImputer()),
    ('scaler', StandardScaler()),
    ('classifier', GaussianProcessClassifier())
])

# Grid search
param_grid = {
    'classifier__n_restarts_optimizer': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'classifier__max_iter_predict': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)

# print best parameters
print(grid.best_params_) # {'classifier__max_iter_predict': 100, 'classifier__n_restarts_optimizer': 0}

# Evaluate model on train and test set
y_train_pred = grid.predict_proba(X_train)
y_test_pred = grid.predict_proba(X_test)

train_auc = roc_auc_score(y_train, y_train_pred[:, 1])
test_auc = roc_auc_score(y_test, y_test_pred[:, 1])

print(f'Train AUC: {train_auc}') # Train AUC: 1
print(f'Test AUC: {test_auc}') # Test AUC: 0.83
