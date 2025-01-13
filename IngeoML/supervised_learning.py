# Copyright 2023 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from scipy.special import expit, softmax
import cvxpy as cp


class ConvexClassifier(ClassifierMixin, BaseEstimator):
    """Convex Classifier
    
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.ensemble import StackingClassifier
    >>> from IngeoML import ConvexClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> svc = LinearSVC()
    >>> forest = RandomForestClassifier()
    >>> convex = ConvexClassifier()
    >>> stack = StackingClassifier([('SVC', svc),
                                     ('Forest', forest)],
                                    final_estimator=convex).fit(X, y)
    >>> stack.final_estimator_.mixer
    array([0.01783184, 0.98216816])
    """

    @property
    def num_classifiers(self):
        """Number of classifiers"""
        return self._num_classifiers

    @num_classifiers.setter
    def num_classifiers(self, value):
        self._num_classifiers = value

    @property
    def num_classes(self):
        """Number of labels"""
        return self.classes.shape[0]

    @property
    def classes(self):
        """Classes"""
        return self._classes

    @classes.setter
    def classes(self, value):
        self._classes = value

    @property
    def mixer(self):
        """Convex combination"""
        return self._mixer

    @mixer.setter
    def mixer(self, value):
        if np.any(value < 0):
            value[value < 0] = 0
            value = value / value.sum()
        self._mixer = value

    @property
    def apply_norm(self):
        """Feature masks"""
        return self._apply_norm

    @apply_norm.setter
    def apply_norm(self, value):
        self._apply_norm = value

    def normalize_2cl(self, X):
        """Compute probabilities in case these are needed"""
        Xs = []
        apply_explit = []
        for x in X.T:
            _norm = False
            if np.any((x < 0) | (x > 1)):
                x = expit(x)
                _norm = True
            Xs.append(x)
            apply_explit.append(_norm)
        self.apply_norm = apply_explit
        return Xs
    
    def fit_2cl(self, X, y, weights):
        """Fit a binary classifier"""
        Xs = self.normalize_2cl(X)
        y_prob = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
        C1 = np.vstack(Xs).T
        coef = cp.Variable(C1.shape[1])
        one = np.ones(C1.shape[1])
        pos = C1 @ coef
        neg = 1 - pos
        hy_prob = cp.vstack([neg, pos]).T
        obj = cp.Minimize(cp.sum(cp.rel_entr(y_prob, hy_prob), axis=1) @ weights)
        constraints = [one @ coef == 1, coef >= 0]
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve()
        except cp.SolverError:
            prob.solve(solver='SCS', time_limit_secs=1800)
        self.mixer = coef.value

    def normalize_kcl(self, X):
        """Compute probabilities in case these are needed"""
        Xs = []
        apply_norm = []
        ncl = self.num_classes
        index = np.arange(0, X.shape[1] + ncl, ncl)
        for strt, stp in zip(index, index[1:]):
            x = X[:, np.arange(strt, stp)]
            _norm = False
            tot = x.sum(axis=1)
            if np.any((tot < 0.999) | (tot > 1.001)):
                x = softmax(x, axis=1)
                _norm = True
            Xs.append(x)
            apply_norm.append(_norm)
        self.apply_norm = apply_norm
        return Xs

    def fit_kcl(self, X, y, weights):
        """Fit a multi-class classifier"""
        C1 = self.normalize_kcl(X)
        y_prob = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
        coef = cp.Variable(len(C1))
        one = np.ones(len(C1))
        pos = C1[0] * coef[0]
        for C, w in zip(C1[1:], coef[1:]):
            pos += C * w

        obj = cp.Minimize(cp.sum(cp.rel_entr(y_prob, pos), axis=1) @ weights)
        constraints = [one @ coef == 1, coef >= 0]
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve()
        except cp.SolverError:
            prob.solve(solver='SCS', time_limit_secs=1800)
        self.mixer = coef.value

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Estimate the parameters given the dataset (`X` and `y`)"""
        self.classes, _ = np.unique(y, return_counts=True)
        elements =  1 / (_ * self.classes.shape[0])
        weights = np.empty(y.shape[0])
        for label, w in zip(self.classes, elements):
            weights[y == label] = w        
        if self.num_classes > 2:
            self.num_classifiers = X.shape[1] // self.num_classes
            assert X.shape[1] % self.num_classes == 0
        else:
            self.num_classifiers = X.shape[1]
        if self.num_classes == 2:
            self.fit_2cl(X, y, weights)
        else:
            self.fit_kcl(X, y, weights)
        return self

    def predict_proba(self, X):
        """Predict class probabilities"""
        if self.num_classes == 2:
            Xs = [expit(x) if flag else x
                for (x, flag) in zip(X.T, self.apply_norm)]
            C1 = np.vstack(Xs).T
            pos = C1 @ self.mixer
            neg = 1 - pos
            return np.c_[neg, pos]
        else:
            Xs = []
            ncl = self.num_classes
            index = np.arange(0, X.shape[1] + ncl, ncl)
            for strt, stp, flag in zip(index, index[1:], self.apply_norm):
                x = X[:, np.arange(strt, stp)]
                if flag:
                    x = softmax(x, axis=1)
                Xs.append(x)
            pos = Xs[0] * self.mixer[0]
            for C, w in zip(Xs[1:], self.mixer[1:]):
                pos += C * w
            return pos            

    def predict(self, X):
        """Predict the classes"""
        hy = self.predict_proba(X)
        return hy.argmax(axis=1)
