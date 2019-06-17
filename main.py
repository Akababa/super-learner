from sklearn import datasets
from sklearn.model_selection import train_test_split

from base import SuperLearner
from sklearn import datasets, linear_model, neighbors, svm, ensemble

ols = linear_model.LinearRegression()
elnet = linear_model.ElasticNetCV(rho=.1)
ridge = linear_model.RidgeCV()
lars = linear_model.LarsCV()
lasso = linear_model.LassoCV()
nn = neighbors.KNeighborsRegressor()
svm1 = svm.SVR(scale_C=True)
svm2 = svm.SVR(kernel='poly', scale_C=True)
rf = ensemble.RandomForestRegressor(n_estimators=10, random_state=1)
model_lib = [ols, elnet, ridge, lars, lasso, nn, svm1, svm2, rf]
model_names = ["OLS", "ElasticNet", "Ridge", "LARS", "LASSO", "kNN", "SVM rbf", "SVM poly", 'RF']

iris = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=0)

# clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf = SuperLearner().fit(X_train, y_train)
print(clf.score(X_test, y_test))
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print(scores)
