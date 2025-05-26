import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def load_data(data_path, features_path):
    data = pd.read_csv(data_path)
    X = pd.read_csv(features_path)
    Y = data['Class'].to_numpy()
    return X, Y

def preprocess(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def split_data(X, Y):
    return train_test_split(X, Y, test_size=0.2, random_state=2)

def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    preds = model.predict(X_test)
    return accuracy_score(Y_test, preds)

def run_models(X_train, X_test, Y_train, Y_test):
    results = {}
    models = {
        'Logistic Regression': LogisticRegression(),
        'SVM': SVC(),
        'Decision Tree': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier()
    }
    for name, model in models.items():
        acc = evaluate_model(model, X_train, Y_train, X_test, Y_test)
        results[name] = acc
    return results
