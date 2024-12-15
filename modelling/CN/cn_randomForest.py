import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer

# Load data
data = pd.read_csv('data/cn_preprocessed.csv')

# Split data
X = data.drop('last_DX', axis=1)
y = data['last_DX']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipe = Pipeline([
    ('imputer', KNNImputer()),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Grid search
param_grid = {
    'classifier__n_estimators': [50, 100, 200, 300, 400, 500],
    'classifier__max_depth': [3, 5, 7, 8, 9, 10, 11, 13],
    'classifier__min_samples_split': [1, 2, 2, 4, 6, 8, 10],
    'classifier__max_features': ['auto', 'sqrt', 'log2']
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)

# print best parameters
print(grid.best_params_)

# Evaluate model on train and test set
y_train_pred = grid.predict_proba(X_train)
y_test_pred = grid.predict_proba(X_test)

train_auc = roc_auc_score(y_train, y_train_pred[:, 1])
test_auc = roc_auc_score(y_test, y_test_pred[:, 1])

print(f'Train AUC: {train_auc}')
print(f'Test AUC: {test_auc}')
