import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_X_y, check_array
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error



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


class BMA():
    def __init__(self, cand_learners=[LinearRegression()]):
        self.cand_learners = cand_learners
        self.weights = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n = len(X)
        self.cand_learners_ = [clone(c) for c in self.cand_learners]
        BIC = np.zeros(len(self.cand_learners))

        self.weights_ = np.exp(-0.5 * BIC) / (sum(-0.5 * BIC))
        return self

    def weights(self):
       return self.weights



if __name__ == "__main__":
    check_estimator(SuperLearner())
