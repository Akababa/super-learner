from sklearn import datasets, linear_model, neighbors, svm, ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from base import SuperLearner
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

v_folds = 5
ols = linear_model.LinearRegression()
elnet = linear_model.ElasticNetCV(l1_ratio=0.5, cv=v_folds, normalize=True)
ridge = linear_model.RidgeCV(cv=v_folds)
lars = linear_model.LarsCV(cv=v_folds, normalize=True)
lasso = linear_model.LassoCV(cv=v_folds, normalize=True)
nn = neighbors.KNeighborsRegressor(weights='distance')
svm1 = svm.SVR(kernel='linear', C=10, gamma='auto')
svm2 = svm.SVR(kernel='poly', C=10, gamma='auto')
rf = ensemble.RandomForestRegressor(n_estimators=50, max_depth=4, min_samples_split=2, random_state=1)
model_lib = [ols, rf ,elnet, ridge, lars, lasso, nn, svm1, svm2]
model_names = ["OLS", "RF", "ElasticNet", "Ridge", "LARS", "LASSO", "kNN", "SVM rbf", "SVM poly"]
meta_learner = ols
diabetes = datasets.load_diabetes()
iris = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.4, random_state=0)

# clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf = SuperLearner(cand_learners=model_lib[0:8],
                  meta_learner=meta_learner,
                   V=v_folds).fit(X_train, y_train)
print(clf.score(X_test, y_test))
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, diabetes.data, diabetes.target, cv=KFold(n_splits=v_folds, random_state=0))
print(scores)
