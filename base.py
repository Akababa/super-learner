import numpy as np
import pandas as pd
from scipy import stats
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

    def fit_cands(self, X, y):
        X, y = check_X_y(X, y)
        # step 1: split data into V blocks
        folds = [i % self.V for i in range(len(X))]
        # step 2-3: train each candidate learner and get CV predictions
        self.cand_learners_ = [clone(c) for c in self.cand_learners]
        Z = [cross_val_predict(cl, X, y, groups=folds, cv=self.V) for cl in self.cand_learners_]
        Z = np.transpose(Z)
        self.Z_train_cv_ = Z
        # step 0: fit candidate learners on whole dataset (have to do this after)
        for cand in self.cand_learners_:
            cand.fit(X, y)

        return Z

    def fit_meta(self, Z, y):
        Z = check_array(Z)
        self.meta_learner_ = clone(self.meta_learner)
        self.meta_learner_.fit(Z, y)

    def fit(self, X, y):
        # steps 0-3: fit candidate learners
        Z = self.fit_cands(X, y)
        # step 4: fit meta learner
        self.fit_meta(Z, y)

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

    def get_cand_stats(self, X1, y1, X2=None, y2=None):
        stuff = []
        for cl, Z_cv in zip(self.cand_learners_, np.transpose(self.Z_train_cv_)):
            stuff.append([type(cl).__name__,
                          mean_squared_error(cl.predict(X1), y1),
                          mean_squared_error(Z_cv, y1)]
                         + ([mean_squared_error(cl.predict(X2), y2)] if X2 is not None else []))
        return stuff

    def get_meta_stats(self, X1, y1, X2=None, y2=None):
        return [f"Meta ({type(self.meta_learner_).__name__})",
                mean_squared_error(self.predict(X1), y1),
                mean_squared_error(self.meta_learner_.predict(self.Z_train_cv_), y1)] \
               + ([mean_squared_error(self.predict(X2), y2)] if X2 is not None else [])

    def debug(self, X1, y1, X2=None, y2=None, skip_fit=False, relative=False):
        """
        Fits on X1, y1 and predicts on X2, y2 and returns a DataFrame of useful info.
        X2, y2 optional.
        """
        X1, y1 = check_X_y(X1, y1)

        test = X2 is not None or y2 is not None
        if test:
            X2, y2 = check_X_y(X2, y2)

        if not skip_fit:
            self.fit(X1, y1)

        stuff = self.get_cand_stats(X1, y1, X2, y2)

        stuff.append(self.get_meta_stats(X1, y1, X2, y2))
        col_names = ["Learner", "Train MSE", "Train CV MSE"] + (["Test MSE"] if test else [])
        df = pd.DataFrame(data=stuff, columns=col_names)
        if relative:
            col_names.remove("Learner")
            col_mins = df[col_names].min(axis=0)
            df[col_names] = df[col_names]/col_mins
        try:
            df["Coefs"] = self.meta_learner_.coef_.tolist() + [sum(self.meta_learner_.coef_)]
        except:  # no coefs
            pass
        if relative:
            df = df.append(pd.Series(), ignore_index=True)
            df.iloc[-1,0] = 'Min Error'
            df.iloc[-1,1:len(col_mins)+1] = col_mins
        df = df.round(4)
        return df


def try_super_learners(cands, metas, X1, y1, X2, y2,relative=False):
    sl = SuperLearner(cand_learners=cands)
    sl.fit_cands(X1, y1)
    Z = sl.Z_train_cv_
    stats = sl.get_cand_stats(X1, y1, X2, y2)
    for meta in metas:
        sl.set_params(meta_learner=meta)
        sl.fit_meta(Z, y1)
        stats.append(sl.get_meta_stats(X1, y1, X2, y2))
    col_names = ["Learner", "Train MSE", "Train CV MSE"] + (["Test MSE"] if X2 is not None else [])
    df = pd.DataFrame(data=stats,
                      columns=col_names)
    if relative:
        col_names.remove("Learner")
        col_mins = df[col_names].min(axis=0)
        df[col_names] = df[col_names] / col_mins
        df = df.append(pd.Series(), ignore_index=True)
        df.iloc[-1, 0] = 'Min Error'
        df.iloc[-1, 1:len(col_mins) + 1] = col_mins
    df = df.round(4)
    return df


class BMA(BaseEstimator, RegressorMixin):
    def __init__(self, cand_learners=[LinearRegression()]):
        self.cand_learners = cand_learners

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n = len(X)
        p = len(np.transpose(X))
        self.cand_learners_ = [clone(c) for c in self.cand_learners]
        k = len(self.cand_learners_)
        BIC = np.zeros(k)
        norm_test = np.zeros((k,2))
        for cand, i in zip(self.cand_learners_, range(k)):
            cand.fit(X, y)
            # https://www.ssc.wisc.edu/~bhansen/718/NonParametrics15.pdf
            e_i = y - cand.predict(X)
            var_e_i = np.var(e_i) # this is hat sigma^2
            norm_test[i,] = stats.shapiro(e_i)
            ll_val = -0.5 * n * np.log(2 * np.pi * var_e_i) - n / 2 # simplified log likelihood
            BIC[i] = -2*ll_val + (p + 2) * np.log(n)

        # using log-sum-exp rule: https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        logsum = max(-0.5*BIC) + np.log(np.sum(np.exp(-0.5*BIC - max(-0.5*BIC))))
        self.weights_ = np.exp(-0.5*BIC - logsum) # weighting formula
        self.norm_test_ = norm_test
        return self

    def weights(self):
        print([type(x).__name__ for x in self.cand_learners_])
        return self.weights_

    def norm_test(self, pval=True):
        if pval:
            return self.norm_test_[:,1]
        else:
            return self.norm_test_[:,0]

    def predict(self, X):
        check_is_fitted(self, 'weights_')
        X = check_array(X)
        y_hat = np.transpose([cl.predict(X) for cl in self.cand_learners_])
        return np.sum(y_hat * self.weights_, axis=1)

    def debug(self, X1, y1, X2=None, y2=None, skip_fit=False, relative=False):
        """
        Fits on X1, y1 and predicts on X2, y2 and returns a DataFrame of useful info.
        X2, y2 optional.
        """
        X1, y1 = check_X_y(X1, y1)

        test = X2 is not None or y2 is not None
        if test:
            X2, y2 = check_X_y(X2, y2)

        if not skip_fit:
            self.fit(X1, y1)

        stuff = []
        for cl in self.cand_learners_:
            stuff.append([type(cl).__name__,
                          mean_squared_error(cl.predict(X1), y1)]
                         + ([mean_squared_error(cl.predict(X2), y2)] if test else []))

        stuff.append(["BMA",
                      mean_squared_error(self.predict(X1), y1)]
                     + ([mean_squared_error(self.predict(X2), y2)] if test else []))

        col_names = ["Learner", "Train MSE"] + (["Test MSE"] if test else [])
        df = pd.DataFrame(data=stuff, columns=col_names)
        if relative:
            col_names.remove("Learner")
            col_mins = df[col_names].min(axis=0)
            df[col_names] = df[col_names]/col_mins
        try:
            df["Coefs"] = self.weights_.tolist() + [sum(self.weights_)]
            df["BIC"] = self.BIC_.tolist() + [None]
        except:  # no coefs
            pass
        if relative:
            df = df.append(pd.Series(), ignore_index=True)
            df.iloc[-1,0] = 'Min Error'
            df.iloc[-1,1:len(col_mins)+1] = col_mins
        df = df.round(4)
        return df

def gbm_summary(gbm, X1, y1, X2=None, y2=None):
    gbm.fit(X1,y1)
    test = X2 is not None or y2 is not None
    if test:
        X2, y2 = check_X_y(X2, y2)
    stuff = []
    stuff.append(["Gradient Boosting",
                  mean_squared_error(gbm.predict(X1), y1)]
                 + ([mean_squared_error(gbm.predict(X2), y2)] if test else []))
    col_names = ["Learner", "Train MSE"]  + (["Test MSE"] if test else [])
    params = pd.DataFrame.from_dict(gbm.best_params_,orient='index')
    print(params)
    df = pd.DataFrame(data=stuff, columns=col_names)
    df = df.round(4)
    return df


if __name__ == "__main__":
    check_estimator(SuperLearner())
