import pathlib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import json


df_train = pd.read_csv(pathlib.Path("../data/interim/bank_train.csv"))
df_test = pd.read_csv(pathlib.Path("../data/interim/bank_test.csv"))


numeric_features = [
    'age',
    'balance',
    'day',
    'campaign',
    'pdays',
    'previous',
]

categorical_features = [
    'job',
    'marital',
    'education',
    'default',
    'housing',
    'loan',
    'contact',
    'month',
    'campaign',
    'pdays',
    'previous',
]


numeric_transformer_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), #impute -> nezinomas reiksmes panaikina idedamas mediana (stulpelio?)
    ('scaler', StandardScaler())])

categorical_transformer_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), #impute -> nezinomas reiksmes panaikina idedami konstanta, nes kategoriniai kintamieji
    ('onehot', OneHotEncoder(handle_unknown='ignore'))]) #ka negautume klaidos 'ignore'

preprocessor_pipe = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_pipe, numeric_features),
        ('cat', categorical_transformer_pipe, categorical_features)])


X_train = df_train.drop('target', axis=1)
y_train = df_train['target']

X_test = df_test.drop('target', axis=1)
y_test = df_test['target']


clf = Pipeline(steps=[
    ('preprocessor', preprocessor_pipe),
    ('classifier', RandomForestClassifier(n_jobs=-1, n_estimators=100))]) #100 medziu

clf.fit(X_train, y_train)


param_grid = {
    'classifier__n_estimators': [5, 10, 30, 50, 100], #lab 3 praplesti parametru gardele
    'classifier__max_depth': [3, 7, 10, 5], #__ u apatiniai bruksneliai sckitlearn kazkas
}

grid_search = GridSearchCV(clf, param_grid, cv=10, iid=False, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

accuracy = ("model accuracy: {:.3f}".format(metrics.accuracy_score(y_test, grid_search.predict(X_test))))

precision = ("model precision: {:.3f}".format(metrics.precision_score(y_test, grid_search.predict(X_test))))

recall = ("model recall: {:.3f}".format(metrics.recall_score(y_test, grid_search.predict(X_test))))

f1 = ("model F1: {:.3f}".format(metrics.f1_score(y_test, grid_search.predict(X_test))))

AuROC = ("model AuROC: {:.3f}".format(metrics.roc_auc_score(y_test, grid_search.predict(X_test))))

metrics = [accuracy, precision, recall, f1, AuROC]

print(metrics)

with open('metrics.json', 'w') as f:
    json.dump(metrics, f)
    
cv_results = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in grid_search.cv_results_.items()}

json.dumps(cv_results, indent=4)

with open('data.json', 'w') as f:
    json.dump(cv_results, f)