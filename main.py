from sklearn import datasets, linear_model, neighbors, svm, ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from base import SuperLearner
from base import BMA
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning)

seed1 = 0
seed2 = 555
v_folds = 5
ols = linear_model.LinearRegression()
elnet = linear_model.ElasticNetCV(l1_ratio=0.5, cv=v_folds, normalize=True)
ridge = linear_model.RidgeCV(cv=v_folds)
lars = linear_model.LarsCV(cv=v_folds, normalize=True)
lasso = linear_model.LassoCV(cv=v_folds, normalize=True)
nn = neighbors.KNeighborsRegressor(weights='distance')
svm1 = svm.SVR(kernel='linear', C=10, gamma='auto')
svm2 = svm.SVR(kernel='poly', C=10, gamma='auto')
rf = ensemble.RandomForestRegressor(n_estimators=50, max_depth=4, min_samples_split=2, random_state=seed1)
model_lib = [ols, rf ,elnet, ridge, lars, lasso, nn, svm1, svm2]
model_names = ["OLS", "RF", "ElasticNet", "Ridge", "LARS", "LASSO", "kNN", "SVM rbf", "SVM poly"]
meta_learner = ols
diabetes = datasets.load_diabetes()
iris = datasets.load_iris()
#print(diabetes.keys())
#pd.DataFrame(data=np.c_[diabetes['target'],diabetes['data']],columns=(['y']+diabetes['feature_names'])).to_csv('diabetes.csv',index=False)
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.4, random_state=seed2)

# clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf = SuperLearner(cand_learners=model_lib[0:8],
                  meta_learner=meta_learner,
                   V=v_folds).fit(X_train, y_train)
bma = BMA(model_lib[1:5])
bma.fit(X_train, y_train)
print(bma.weights())
print(bma.score(X_test, y_test))

print(clf.score(X_test, y_test))
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, diabetes.data, diabetes.target, cv=KFold(n_splits=v_folds, random_state=seed2))
print(scores)
print(np.mean(scores))

scores_bma = cross_val_score(bma, diabetes.data, diabetes.target, cv=KFold(n_splits=v_folds, random_state=seed2))
print(scores_bma)
print(np.mean(scores_bma))
