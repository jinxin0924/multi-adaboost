# Multi-adaboost
Python implementation of multi-class adaboost decision trees.

References:

1. [multi-adaboost](https://web.stanford.edu/~hastie/Papers/samme.pdf)

2. [scikit-learn:weight_boosting](https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/ensemble/weight_boosting.py#L289)

3. Machine Learning in Action, Chapter 1.3-decision trees

# Learning experience
This is my learning experience when I read the paper [Multi-class AdaBoost](https://web.stanford.edu/~hastie/Papers/SII-2-3-A8-Zhu.pdf). To learn this new algorithm, first I reviewed **AdaBoost** and its relationship with  **Forward Stagewise Additive Modeling**. Then I realized that multi-AdaBoost is natural extension of AdaBoost.

## AdaBoost
Let T(x) denote a weak multi-class classifier that assigns a class label to x, then the AdaBoost algorithm proceeds as follows:
![AdaBoost](http://upload-images.jianshu.io/upload_images/1825085-234a26e11524427e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Pay attention to the weak classifier weight, α, to let α >0, we need to ensure the err<0.5. This can be easily done when it's binary classification problem, because error rate of random guessing is 0.5. But  it is difficult to achieve in multi-class case. So it may easily fail in multi-class case.

## Forward Stagewise Additive Modeling

First let's take a look at **Additive Model**:

![Additive Model](http://upload-images.jianshu.io/upload_images/1825085-5c0556e913dc8425.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Typically, {βm, γm} are estimated by minimizing some loss function L, which measures the prediction errors over training data.
![](http://upload-images.jianshu.io/upload_images/1825085-c908c9e7b2ad6e04.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
Directly optimizing such loss function is often difficult. However, consider it's an additive model, a simple greedy method can be used. We can  sequentially optimize the following  loss function, then add new base functions to the expansion function f(x) without changing the parameters that have been added.

![loss function for each step](http://upload-images.jianshu.io/upload_images/1825085-f17795a2453e0199.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Thus the Forward Stagewise Additive Modeling algorithm (denote FSAM) is:
1. Initialize f_0(x)=0
2. For *m* = 1 to *M* :
  a) Compute
![](http://www.forkosh.com/mathtex.cgi? \Large  \ (\beta_m,\gamma_m) = \mathop{\arg\max}_{\beta, \gamma} \sum_{i=1}^{N} L[y_i,f_{m-1}(x_i) + \beta b(x_i;\gamma)]  (2.1)  )

  b) Set 
![](http://www.forkosh.com/mathtex.cgi? \Large  f_m(x) = f_{m-1}(x) + \beta_m b(x;\gamma_m)    --(2.2))

## AdaBoost and Forward Stagewise Additive Modeling
In fact AdaBoost is equivalent to Forward Stagewise Additive Modeling using the exponential loss funtion
![](http://www.forkosh.com/mathtex.cgi? \Large  L[y,f(x)] = exp[-yf(x)]. )

Again let T(x) denote a classifier that assigns a class label to x, in this case, T(x)=1 or -1.

Then (2.1) is:
![](http://www.forkosh.com/mathtex.cgi? \Large  (\beta_m,T_m) =  \mathop{\arg\max}_{\beta, T} \sum_{i=1}^{N} \ exp [-y_i[f_{m-1}(x_i) + \beta T(x_i)]]) 

To simplify the expression, let 
![](http://www.forkosh.com/mathtex.cgi? \Large  w^{(m)}_i = exp[-y_i f_{m-1}(x_i)] ) 
Notice that w depends neither on β nor T(x). 

Then the expression is
![](http://www.forkosh.com/mathtex.cgi? \Large  (\beta_m,T_m) =  \mathop{\arg\max}_{\beta, T} \sum_{i=1}^{N} w^{(m)}_i exp [-\beta T(x_i)])

It can be rewritten as
![](http://www.forkosh.com/mathtex.cgi? \Large  (\beta_m,T_m) =  \mathop{\arg\max}_{\beta, T} \ e^{-\beta}\sum_{y_i = T(x_i)} w^{(m)}_i  + e^{\beta}\sum_{y_i \neq T(x_i)} w^{(m)}_i \\  =  \mathop{\arg\max}_{\beta, T} \ (e^\beta - e^{-\beta}) \sum_{i=1}^N w_i^{(m)} I[y_i \neq T(x_i)] + e^{-\beta}\sum_{i=1}^N w_i^{(m)} )

Now we can get that 
![](http://www.forkosh.com/mathtex.cgi? \Large \beta _m=\frac{1}{2} log \frac{\sum_{i=1}^N w_i^{(m)} (1- I[y_i \neq T_m(x_i)]}{\sum_{i=1}^N I[y_i \neq T_m(x_i)]} )

denote **err_m** as
![](http://www.forkosh.com/mathtex.cgi? \Large err_m = \frac {\sum_{i=1}^N w_i^{(m)} I[y_i \neq T_m(x_i) ]} {\sum_{i=1}^N w_i^{(m)}} )

Then 
![](http://www.forkosh.com/mathtex.cgi? \Large \beta _m=\frac{1}{2} log \frac{1-err_m}{err_m})

Look familiar, right? It is actually the AdaBoost's weak classifier weight, α_m, except the 1/2. I take the 1/2 as learning rate.

Also recall w_m,
![](http://www.forkosh.com/mathtex.cgi? \Large  w^{(m)}_i = exp[-y_i f_{m-1}(x_i)] )
Then 
![](http://www.forkosh.com/mathtex.cgi? \Large  w_i^{(m+1)}_i = w_i^{(m)} exp[-\beta_m y_i T_m(x_i)] \\= w_i^{(m)} \cdot exp[2\beta_m I[y_i \neq T_m(x_i)] \cdot exp(-\beta_m))
using the fact that 
![](http://www.forkosh.com/mathtex.cgi? \Large  -y_iT_m(x_i)=2 \cdot I[ y_i \neq T_m(x_i)]-1)
compare to the weight in AdaBoost,

![weight in AdaBoost](http://upload-images.jianshu.io/upload_images/1825085-220c794288c8993a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
We can know that they are equivalent.


As I mentioned before, AdaBoost can easily fail in multi-class case due to the error rate. To avoid this, a natural way is to do something about α. That's what *SAMME* does. SAMME is short for Stagewise Additive Modeling using a Multi-class Exponential loss function. 

**SAMME**
![](http://upload-images.jianshu.io/upload_images/1825085-4ddd1ee392f1ad59.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](http://upload-images.jianshu.io/upload_images/1825085-f56688d8bc6e5e13.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
Note that SAMME  is very similar to AdaBoost, except the term log(K-1). Obviously, when K=2, SAMME is equivalent to AdaBoost. In SAMME, in order for α to be positive, we only need err<(K-1)/K. It's just a little better than random guessing.

## SAMME and Forward Stagewise Additive Modeling
We can mimic the steps in **AdaBoost and Forward Stagewise Additive Modeling** to prove SAMME is equivalent to Forward Stagewise Additive Modeling using a Multi-class Exponential loss function. 

In the multi-class classification setting, we can recode the output c with a K-dimensional vector y, as:
y_k = 1 if c == k else -1 / (K - 1), like 
![y encoding](http://upload-images.jianshu.io/upload_images/1825085-31b67055367841f2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Given the training data, we wish to find f(x) = (f1(x),...,fK(x))T such that
![](http://www.forkosh.com/mathtex.cgi? \Large  \mathop{\min}_{f(x)}\sum_{i=1}^nL[y_i,f(x_i)]\\subject\ to \quad  f_1(x)+f_2(x)+...+f_K(X)=0 ) 
We consider f(x) has following form:
![](http://www.forkosh.com/mathtex.cgi? \Large  f(x) = \sum_{m=1}^M\beta_{m}g_{m}(x) )
where β is weight and g(x) is base classifier.

Let's start from the loss function. The multi-class exponential loss function:

![](http://www.forkosh.com/mathtex.cgi? \Large  L(y,f)=exp[-1/K(y_1f_1+...+y_Kf_K)]\\=exp(-\frac{1}{K} y^Tf) )

Then (2.1) in multi-class case is:
![](http://www.forkosh.com/mathtex.cgi? \Large  (\beta_m,g_m) =  \mathop{\arg\max}_{\beta, g} \sum_{i=1}^{N} \ exp [-\frac{1}{K} y_i^T(f_{m-1}(x_i) + \beta g(x_i)]  \\ =\mathop{\arg\max}_{\beta, g} \sum_{i=1}^{N}w_i\ exp[-\frac{1}{K}\beta y_i^T g(x_i)] \qquad\qquad(3.1)) 
where 
![](http://www.forkosh.com/mathtex.cgi? \Large w_i^m=exp[-\frac{1}{K}\beta y_i^Tf_{m-1}(x_i)] )
Notice that every g(x) has a one-to-one correspondence with a multi-class classifier T (x) in the following way:
![](http://www.forkosh.com/mathtex.cgi? \Large T(x) =k,\qquad if \quad g_K(x)=1 )

Hence, solving for g_m(x)  is equivalent to finding the multi-class classifier T_m(x) that can generate g_m(x).

Recall that in AdaBoost and Forward Stagewise Additive Modeling, we rewrite the expression as
![](http://www.forkosh.com/mathtex.cgi? \Large  (\beta_m,T_m) =  \mathop{\arg\max}_{\beta, T} \ e^{-\beta}\sum_{y_i = T(x_i)} w^{(m)}_i  + e^{\beta}\sum_{y_i \neq T(x_i)} w^{(m)}_i \\  =  \mathop{\arg\max}_{\beta, T} \ (e^\beta - e^{-\beta}) \sum_{i=1}^N w_i^{(m)} I[y_i \neq T(x_i)] + e^{-\beta}\sum_{i=1}^N w_i^{(m)} )

Similarly, we rewrite (3.1) as
![](http://www.forkosh.com/mathtex.cgi? \Large  (\beta_m,T_m) =  \mathop{\arg\max}_{\beta, T} \ e^{-\frac{\beta}{K-1}}\sum_{c_i = T(x_i)} w^{(m)}_i  + e^{\frac{\beta}{(K-1)^2}}\sum_{c_i \neq T(x_i)} w^{(m)}_i \\  =  \mathop{\arg\max}_{\beta, T} \ [e^{\frac{\beta}{(K-1)^2}} - e^{-\frac{\beta}{K-1}} ]\sum_{i=1}^N w_i^{(m)} I[c_i \neq T(x_i)]+ e^{-\frac{\beta}{K-1}}\sum_{i=1}^N w_i^{(m)} )

So the solution is
![](http://www.forkosh.com/mathtex.cgi? \Large \beta _m=\frac{(K-1)^2}{K} [log \frac{1-err_m}{err_m} +log(K-1) ] ) 
where error
![](http://www.forkosh.com/mathtex.cgi? \Large err_m = \frac {\sum_{i=1}^N w_i^{(m)} I[c_i \neq T_m(x_i) ]} {\sum_{i=1}^N w_i^{(m)}} ).

Look at the β_m, we can ignore (K-1)^2/K, the rest is just the difference between AdaBoost and SAMME. It proves that the extra term log(K − 1) in SAMME is not artificial. 

## SAMME.R
The SAMME algorithm expects the weak learner to deliver a classifier T (x) ∈ {1, . . . , K}. Analternative is to use real-valued confidence-rated predictions such as weighted probability estimates,to update the additive model, rather than the classifications themselves.
![](http://upload-images.jianshu.io/upload_images/1825085-d39bab8d151d2431.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![](http://upload-images.jianshu.io/upload_images/1825085-0b4d10df6206fe19.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
In my opinion, the SAMME.R uses more information than SAMME, and has more robustness.

## Python3 Implementation of Multi-class AdaBoost using SAMME and SAMME.R
```
__author__ = 'Xin'

import numpy as np
from numpy.core.umath_tests import inner1d
from copy import deepcopy


class AdaBoostClassifier(object):
    '''
    Parameters
    -----------
    base_estimator: object
        The base model from which the boosted ensemble is built.

    n_estimators: integer, optional(default=50)
        The maximum number of estimators

    learning_rate: float, optional(default=1)

    algorithm: {'SAMME','SAMME.R'}, optional(default='SAMME.R')
        SAMME.R uses predicted probabilities to update wights, while SAMME uses class error rate

    random_state: int or None, optional(default=None)


    Attributes
    -------------
    estimators_: list of base estimators

    estimator_weights_: array of floats
        Weights for each base_estimator

    estimator_errors_: array of floats
        Classification error for each estimator in the boosted ensemble.
    
    Reference:
    1. [multi-adaboost](https://web.stanford.edu/~hastie/Papers/samme.pdf)

    2. [scikit-learn:weight_boosting](https://github.com/scikit-learn/
    scikit-learn/blob/51a765a/sklearn/ensemble/weight_boosting.py#L289)
    
    '''

    def __init__(self, *args, **kwargs):
        if kwargs and args:
            raise ValueError(
                '''AdaBoostClassifier can only be called with keyword
                   arguments for the following keywords: base_estimator ,n_estimators,
                    learning_rate,algorithm,random_state''')
        allowed_keys = ['base_estimator', 'n_estimators', 'learning_rate', 'algorithm', 'random_state']
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise ValueError(keyword + ":  Wrong keyword used --- check spelling")

        n_estimators = 50
        learning_rate = 1
        algorithm = 'SAMME.R'
        random_state = None

        if kwargs and not args:
            if 'base_estimator' in kwargs:
                base_estimator = kwargs.pop('base_estimator')
            else:
                raise ValueError('''base_estimator can not be None''')
            if 'n_estimators' in kwargs: n_estimators = kwargs.pop('n_estimators')
            if 'learning_rate' in kwargs: learning_rate = kwargs.pop('learning_rate')
            if 'algorithm' in kwargs: algorithm = kwargs.pop('algorithm')
            if 'random_state' in kwargs: random_state = kwargs.pop('random_state')

        self.base_estimator_ = base_estimator
        self.n_estimators_ = n_estimators
        self.learning_rate_ = learning_rate
        self.algorithm_ = algorithm
        self.random_state_ = random_state
        self.estimators_ = list()
        self.estimator_weights_ = np.zeros(self.n_estimators_)
        self.estimator_errors_ = np.ones(self.n_estimators_)


    def _samme_proba(self, estimator, n_classes, X):
        """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].

        References
        ----------
        .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

        """
        proba = estimator.predict_proba(X)

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
        log_proba = np.log(proba)

        return (n_classes - 1) * (log_proba - (1. / n_classes)
                                  * log_proba.sum(axis=1)[:, np.newaxis])


    def fit(self, X, y):
        self.n_samples = X.shape[0]
        # There is hidden trouble for classes, here the classes will be sorted.
        # So in boost we have to ensure that the predict results have the same classes sort
        self.classes_ = np.array(sorted(list(set(y))))
        self.n_classes_ = len(self.classes_)
        for iboost in range(self.n_estimators_):
            if iboost == 0:
                sample_weight = np.ones(self.n_samples) / self.n_samples

            sample_weight, estimator_weight, estimator_error = self.boost(X, y, sample_weight)

            # early stop
            if estimator_error == None:
                break

            # append error and weight
            self.estimator_errors_[iboost] = estimator_error
            self.estimator_weights_[iboost] = estimator_weight

            if estimator_error <= 0:
                break

        return self


    def boost(self, X, y, sample_weight):
        if self.algorithm_ == 'SAMME':
            return self.discrete_boost(X, y, sample_weight)
        elif self.algorithm_ == 'SAMME.R':
            return self.real_boost(X, y, sample_weight)

    def real_boost(self, X, y, sample_weight):
        estimator = deepcopy(self.base_estimator_)
        if self.random_state_:
            estimator.set_params(random_state=1)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_pred = estimator.predict(X)
        incorrect = y_pred != y
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)

        # if worse than random guess, stop boosting
        if estimator_error >= 1.0 - 1 / self.n_classes_:
            return None, None, None

        y_predict_proba = estimator.predict_proba(X)
        # repalce zero
        y_predict_proba[y_predict_proba < np.finfo(y_predict_proba.dtype).eps] = np.finfo(y_predict_proba.dtype).eps

        y_codes = np.array([-1. / (self.n_classes_ - 1), 1.])
        y_coding = y_codes.take(self.classes_ == y[:, np.newaxis])

        # for sample weight update
        intermediate_variable = (-1. * self.learning_rate_ * (((self.n_classes_ - 1) / self.n_classes_) *
                                                              inner1d(y_coding, np.log(
                                                                  y_predict_proba))))  #dot iterate for each row

        # update sample weight
        sample_weight *= np.exp(intermediate_variable)

        sample_weight_sum = np.sum(sample_weight, axis=0)
        if sample_weight_sum <= 0:
            return None, None, None

        # normalize sample weight
        sample_weight /= sample_weight_sum

        # append the estimator
        self.estimators_.append(estimator)

        return sample_weight, 1, estimator_error


    def discrete_boost(self, X, y, sample_weight):
        estimator = deepcopy(self.base_estimator_)
        if self.random_state_:
            estimator.set_params(random_state=1)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_pred = estimator.predict(X)
        incorrect = y_pred != y
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)

        # if worse than random guess, stop boosting
        if estimator_error >= 1 - 1 / self.n_classes_:
            return None, None, None

        # update estimator_weight
        estimator_weight = self.learning_rate_ * np.log((1 - estimator_error) / estimator_error) + np.log(
            self.n_classes_ - 1)

        if estimator_weight <= 0:
            return None, None, None

        # update sample weight
        sample_weight *= np.exp(estimator_weight * incorrect)

        sample_weight_sum = np.sum(sample_weight, axis=0)
        if sample_weight_sum <= 0:
            return None, None, None

        # normalize sample weight
        sample_weight /= sample_weight_sum

        # append the estimator
        self.estimators_.append(estimator)

        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None

        if self.algorithm_ == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            pred = sum(self._samme_proba(estimator, n_classes, X) for estimator in self.estimators_)
        else:  # self.algorithm == "SAMME"
            pred = sum((estimator.predict(X) == classes).T * w
                       for estimator, w in zip(self.estimators_,
                                               self.estimator_weights_))

        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            pred = pred.sum(axis=1)
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)


    def predict_proba(self, X):
        if self.algorithm_ == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            proba = sum(self._samme_proba(estimator, self.n_classes_, X)
                        for estimator in self.estimators_)
        else:  # self.algorithm == "SAMME"
            proba = sum(estimator.predict_proba(X) * w
                        for estimator, w in zip(self.estimators_,
                                                self.estimator_weights_))

        proba /= self.estimator_weights_.sum()
        proba = np.exp((1. / (n_classes - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba
```