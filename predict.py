import numpy as np
import pandas as pd

from estimators.ridge_classifier_estimator import RidgeClassifier
from estimators.ridge_regression_estimator import RidgeRegression
from estimators.svm_estimator import SVM
from train import arg_parser


def make_preds(args, X_train, y_train, X_test):
    if args.model == "svm":
        model = SVM(kernel=args.kernel, alpha=args.alpha, m=args.m, k=args.k)
    elif args.model == "ridge_classifier":
        model = RidgeClassifier(kernel=args.kernel, alpha=args.alpha, m=args.m, k=args.k)
    elif args.model == "ridge_regression":
        model = RidgeRegression(kernel=args.kernel, alpha=args.alpha, m=args.m, k=args.k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds


def predict(args):
    if args.data_type == "float":
        Xtr0, Xtr1, Xtr2 = pd.read_csv("data/Xtr0_mat100.csv", header=None, delimiter=" ").astype(float).values, \
                           pd.read_csv("data/Xtr1_mat100.csv", header=None, delimiter=" ").astype(float).values, \
                           pd.read_csv("data/Xtr2_mat100.csv", header=None, delimiter=" ").astype(float).values
        Xte0, Xte1, Xte2 = pd.read_csv("data/Xte0_mat100.csv", header=None, delimiter=" ").astype(float).values, \
                           pd.read_csv("data/Xte1_mat100.csv", header=None, delimiter=" ").astype(float).values, \
                           pd.read_csv("data/Xte2_mat100.csv", header=None, delimiter=" ").astype(float).values

    elif args.data_type == "string":
        Xtr0, Xtr1, Xtr2 = pd.read_csv("data/Xtr0.csv", index_col="Id").values[:, 0], \
                           pd.read_csv("data/Xtr1.csv", index_col="Id").values[:, 0], \
                           pd.read_csv("data/Xtr2.csv", index_col="Id").values[:, 0]
        Xte0, Xte1, Xte2 = pd.read_csv("data/Xte0.csv", index_col="Id").values[:, 0], \
                           pd.read_csv("data/Xte1.csv", index_col="Id").values[:, 0], \
                           pd.read_csv("data/Xte2.csv", index_col="Id").values[:, 0]
    Ytr0, Ytr1, Ytr2 = pd.read_csv("data/Ytr0.csv", index_col="Id").replace(0, -1).values[:, 0], \
                       pd.read_csv("data/Ytr1.csv", index_col="Id").replace(0, -1).values[:, 0], \
                       pd.read_csv("data/Ytr2.csv", index_col="Id").replace(0, -1).values[:, 0]
    preds0 = make_preds(args, Xtr0, Ytr0, Xte0)
    preds1 = make_preds(args, Xtr1, Ytr1, Xte1)
    preds2 = make_preds(args, Xtr2, Ytr2, Xte2)
    preds = np.concatenate([preds0, preds1, preds2]).astype(int)
    preds_df = pd.DataFrame(data=preds, index=[k * 1000 + i for k in range(3) for i in range(1000)],
                            columns=["Bound"]).replace(-1, 0)
    preds_df.to_csv(f"submissions/{args.model}-{args.kernel}-{args.alpha}-{args.m}-{args.k}.csv", index_label="Id")


if __name__ == '__main__':
    args = arg_parser()
    predict(args)