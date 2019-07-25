import numpy as np
import scipy as sp
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


class BMA(BaseEstimator, RegressorMixin):
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
            mse_eps = mean_squared_error(y, cand.predict(X))
            ll_val = -n*np.log(2*np.pi*mse_eps)/2 - n*mse_eps/(2*mse_eps)
            BIC[i] = ll_val + (p+2)*np.log(n)
            #BIC[i] = np.log(n*mean_squared_error(y, cand.predict(X))) + (p+2)*np.log(n)

        self.weights_ = np.exp(-0.5 * BIC) / (sum(np.exp(-0.5 * BIC)))
        return self

    def weights(self):
        print([type(x).__name__ for x in self.cand_learners_])
        return self.weights_

    def predict(self, X):
        check_is_fitted(self, 'cand_learners')
        X = check_array(X)
        y_hat = np.transpose([cl.predict(X) for cl in self.cand_learners_])
        return np.sum(y_hat*self.weights_, axis=1)


if __name__ == "__main__":
    check_estimator(SuperLearner())
