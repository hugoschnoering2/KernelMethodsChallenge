import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from estimators.logistic_ridge_estimator import KernelLogRidgeEstimator


def train(args):
    if args.data_type == "float":
        Xtr0, Xtr1, Xtr2 = pd.read_csv("data/Xtr0_mat100.csv", header=None, delimiter=" ").astype(float), \
                           pd.read_csv("data/Xtr1_mat100.csv", header=None, delimiter=" ").astype(float), \
                           pd.read_csv("data/Xtr2_mat100.csv", header=None, delimiter=" ").astype(float)
    elif args.data_type == "string":
        Xtr0, Xtr1, Xtr2 = pd.read_csv("data/Xtr0.csv", index_col="Id"), \
                           pd.read_csv("data/Xtr1.csv", index_col="Id"), \
                           pd.read_csv("data/Xtr2.csv", index_col="Id")
    X = pd.concat([Xtr0, Xtr1, Xtr2], axis=0).values
    Ytr0, Ytr1, Ytr2 = pd.read_csv("data/Ytr0.csv", index_col="Id"), \
                       pd.read_csv("data/Ytr1.csv", index_col="Id"), \
                       pd.read_csv("data/Ytr2.csv", index_col="Id")
    y = pd.concat([Ytr0, Ytr1, Ytr2], ignore_index=True, axis=0).values

    kf = KFold(args.folds, random_state=0, shuffle=True)
    scores = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {i + 1}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index].reshape(-1), y[test_index].reshape(-1)
        if args.model == "logistic_regression":
            model = KernelLogRidgeEstimator(args.kernel, args.alpha)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(preds)
        print(y_test)
        score = np.sum(preds == y_test) / len(preds)
        print(f"Val accuracy: {score}")
        scores.append(score)
    print(f"Average accuracy over folds: {np.mean(scores)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="logistic_regression", choices=["logistic_regression"])
    parser.add_argument("-k", "--kernel", type=str, default="spectrum")
    parser.add_argument("-f", "--folds", type=int, default=4)
    parser.add_argument("-a", "--alpha", type=float, default=0.1)
    parser.add_argument("-t", "--data_type", type=str, choices=['string', 'float'], default="string")
    args = parser.parse_args()
    train(args)