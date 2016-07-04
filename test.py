__author__ = 'Xin'
'''
Reference:
Multi-class AdaBoosted Decision Trees:
http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html
'''

import matplotlib.pyplot as plt

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


X, y = make_gaussian_quantiles(n_samples=13000, n_features=10,
                               n_classes=3, random_state=1)

n_split = 3000

X_train, X_test = X[:n_split], X[n_split:]
y_train, y_test = y[:n_split], y[n_split:]

bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1)


bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME")


bdt_real.fit(X_train, y_train)
bdt_discrete.fit(X_train, y_train)



n_trees_discrete = len(bdt_discrete)
n_trees_real = len(bdt_real)

from multi_AdaBoost import AdaBoostClassifier as Ada

bdt_real_test = Ada(
    base_estimator=DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1)
bdt_real_test.fit(X_train, y_train)

bdt_discrete_test = Ada(
    base_estimator=DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1,
    algorithm='SAMME')
bdt_discrete_test.fit(X_train, y_train)


discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
test_real_errors=bdt_real_test.estimator_errors_[:]
test_discrete_errors=bdt_discrete_test.estimator_errors_[:]

plt.figure(figsize=(15, 5))
plt.subplot(221)
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,
         "b", label='SAMME', alpha=.5)
plt.legend()
plt.ylabel('Error')
plt.xlabel('Number of Trees')
plt.ylim((.2,
         max(real_estimator_errors.max(),
             discrete_estimator_errors.max()) * 1.2))
plt.xlim((-20, len(bdt_discrete) + 20))

plt.subplot(222)
plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
         "r", label='SAMME.R', alpha=.5,color='r')
plt.legend()
plt.ylabel('Error')
plt.xlabel('Number of Trees')
plt.ylim((.2,
         max(real_estimator_errors.max(),
             discrete_estimator_errors.max()) * 1.2))
plt.xlim((-20, len(bdt_discrete) + 20))

plt.subplot(224)
plt.plot(range(1, n_trees_real + 1), test_real_errors,
         "r", label='test_real', alpha=.5, color='b')

plt.legend()
plt.ylabel('Error')
plt.xlabel('Number of Trees')
plt.ylim((.2,
         max(real_estimator_errors.max(),
             discrete_estimator_errors.max()) * 1.2))
plt.xlim((-20, len(bdt_discrete) + 20))

plt.subplot(223)
plt.plot(range(1, n_trees_real + 1), test_discrete_errors,
         "r", label='test_discrete', alpha=.5)

plt.legend()
plt.ylabel('Error')
plt.xlabel('Number of Trees')
plt.ylim((.2,
         max(real_estimator_errors.max(),
             discrete_estimator_errors.max()) * 1.2))
plt.xlim((-20, len(bdt_discrete) + 20))


from sklearn.metrics import accuracy_score

print(accuracy_score(bdt_real.predict(X_test),y_test))
print(accuracy_score(bdt_real_test.predict(X_test),y_test))
print(accuracy_score(bdt_discrete.predict(X_test),y_test))
print(accuracy_score(bdt_discrete_test.predict(X_test),y_test))
