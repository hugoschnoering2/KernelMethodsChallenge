import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from estimators.svm_estimator import SVM


def cross_validate(args, X, y):
    kf = KFold(args.folds, random_state=0, shuffle=True)
    scores = []

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {i + 1}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index].reshape(-1), y[test_index].reshape(-1)
        model = SVM(kernel=args.kernel, alpha=args.alpha, m=args.m, k=args.k)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"Val accuracy: {score}")
        scores.append(score)
    print(f"Average accuracy over folds: {np.mean(scores)}")
    return np.mean(scores)


def train(args):
    if args.data_type == "float":
        Xtr0, Xtr1, Xtr2 = pd.read_csv("data/Xtr0_mat100.csv", header=None, delimiter=" ").astype(float).values, \
                           pd.read_csv("data/Xtr1_mat100.csv", header=None, delimiter=" ").astype(float).values, \
                           pd.read_csv("data/Xtr2_mat100.csv", header=None, delimiter=" ").astype(float).values
    elif args.data_type == "string":
        Xtr0, Xtr1, Xtr2 = pd.read_csv("data/Xtr0.csv", index_col="Id").values[:, 0], \
                           pd.read_csv("data/Xtr1.csv", index_col="Id").values[:, 0], \
                           pd.read_csv("data/Xtr2.csv", index_col="Id").values[:, 0]
    Ytr0, Ytr1, Ytr2 = pd.read_csv("data/Ytr0.csv", index_col="Id").replace(0, -1).values[:, 0], \
                       pd.read_csv("data/Ytr1.csv", index_col="Id").replace(0, -1).values[:, 0], \
                       pd.read_csv("data/Ytr2.csv", index_col="Id").replace(0, -1).values[:, 0]

    print("## DATASET 1 ##")
    avg1 = cross_validate(args, Xtr0, Ytr0)
    print("## DATASET 2 ##")
    avg2 = cross_validate(args, Xtr1, Ytr1)
    print("## DATASET 3 ##")
    avg3 = cross_validate(args, Xtr2, Ytr2)
    print(f"Average score over all dataset:{np.mean([avg1, avg2, avg3])}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--m", type=int, default=0)
    parser.add_argument("-k", "--k", type=int, default=6)
    parser.add_argument("-kn", "--kernel", type=str, default="mismatch", choices=["mismatch"])
    parser.add_argument("-f", "--folds", type=int, default=4)
    parser.add_argument("-a", "--alpha", type=float, default=0.1)
    parser.add_argument("-t", "--data_type", type=str, choices=['string', 'float'], default="string")
    args = parser.parse_args()
    train(args)
