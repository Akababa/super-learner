import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_X_y, check_array
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted


class SuperLearner(BaseEstimator, RegressorMixin):
    def __init__(self, cand_learners=(LinearRegression(), RandomForestRegressor(n_estimators=10, random_state=1)),
                 meta_learner=LinearRegression(),
                 V=3):
        self.cand_learners = cand_learners
        self.meta_learner = meta_learner
        self.V = V

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        # step 1: split data into V blocks
        n = len(X)
        folds = [i % self.V for i in range(n)]
        # step 2-3: train each candidate learner and get CV predictions
        self.cand_learners_ = [clone(c) for c in self.cand_learners]
        Z = [cross_val_predict(cl, X, y, groups=folds, cv=self.V) for cl in self.cand_learners_]
        Z = np.transpose(Z)
        Z = check_array(Z)
        # step 4: train meta learner
        self.meta_learner_ = clone(self.meta_learner)
        self.meta_learner_.fit(Z, y)

        # step 0: fit candidate learners on whole dataset (have to do this after)
        for cand in self.cand_learners_:
            cand.fit(X, y)

        return self

    def predict(self, X):
        check_is_fitted(self, "meta_learner_")
        X = check_array(X)
        Z = [cl.predict(X) for cl in self.cand_learners_]
        Z = np.transpose(Z)
        return self.meta_learner_.predict(Z)

    def mse(self, X, y):
        X, y = check_X_y(X, y)
        return mean_squared_error(y, self.predict(X))

    def debug(self, X1, y1, X2, y2):
        X1, y1 = check_X_y(X1, y1)
        X2, y2 = check_X_y(X2, y2)
        stuff = []
        for cl in self.cand_learners_:
            stuff.append([type(cl).__name__, mean_squared_error(cl.predict(X1), y1),
                          mean_squared_error(cl.predict(X2), y2)])

        stuff.append(["meta", mean_squared_error(self.predict(X1), y1),
                      mean_squared_error(self.predict(X2), y2)])

        df = pd.DataFrame(data=stuff, columns=["name", "training mse", "testing mse"])
        if type(self.meta_learner_).__name__ == "LinearRegression":
            df["coef"] = self.meta_learner_.coef_.tolist() + [None]

        return df


class BMA():
    def __init__(self, cand_learners=[LinearRegression()]):
        self.cand_learners = cand_learners
        self.weights_ = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n = len(X)
        p = len(np.transpose(X))
        self.cand_learners_ = [clone(c) for c in self.cand_learners]
        k = len(self.cand_learners_)
        BIC = np.zeros(k)
        for cand, i in zip(self.cand_learners_, range(k)):
            cand.fit(X, y)
            BIC[i] = np.log(k * mean_squared_error(y, cand.predict(X))) + (p + 2) * np.log(n)

        self.weights_ = np.exp(-0.5 * BIC) / (sum(np.exp(-0.5 * BIC)))
        return self

    def weights(self):
        print([type(x).__name__ for x in self.cand_learners_])
        return self.weights_


if __name__ == "__main__":
    check_estimator(SuperLearner())
