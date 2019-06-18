from sklearn import datasets, linear_model, neighbors, svm, ensemble
from sklearn.model_selection import train_test_split

from base import SuperLearner

v_folds = 5
ols = linear_model.LinearRegression()
elnet = linear_model.ElasticNetCV(l1_ratio=0.5, cv=v_folds)
ridge = linear_model.RidgeCV(cv=v_folds)
lars = linear_model.LarsCV(cv=v_folds)
lasso = linear_model.LassoCV(cv=v_folds)
nn = neighbors.KNeighborsRegressor()
svm1 = svm.SVR(kernel='linear', C=10, gamma='auto')
svm2 = svm.SVR(kernel='poly', C=10, gamma='auto')
rf = ensemble.RandomForestRegressor(n_estimators=20, random_state=1)
model_lib = [ols, rf ,elnet, ridge, lars, lasso, nn, svm1, svm2]
model_names = ["OLS", "RF", "ElasticNet", "Ridge", "LARS", "LASSO", "kNN", "SVM rbf", "SVM poly"]
meta_learner = ols
iris = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=0)

# clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf = SuperLearner(cand_learners=model_lib[0:8],
                  meta_learner=meta_learner).fit(X_train, y_train)
print(clf.score(X_test, y_test))
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, iris.data, iris.target, cv=v_folds)
print(scores)
