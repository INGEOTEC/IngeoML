# Copyright 2024 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from joblib import Parallel, delayed
import numpy as np
from IngeoML.utils import progress_bar


def feature_importance(y, hy, predictions,
                       score=None, n_jobs: int=1):
    """Estimate the feature importance of the model"""
    def compute_score(y, i):
        return [score(y, j) for j in i]

    if score is None:
        score = lambda y, hy: f1_score(y, hy, average='macro')
    base = score(y, hy)
    hy = Parallel(n_jobs=n_jobs)(delayed(compute_score)(y, i)
                                 for i in progress_bar(predictions))
    hy = np.array(hy)
    return base - hy


def predict_shuffle_inputs(model, X, times: int=100, n_jobs: int=1):
    """Predict X by shuffling all the inputs"""
    def predict(model, X, rng, i):
        inner = []
        for _ in range(times):
            rng.shuffle(X[:, i])
            inner.append(model.predict(X))
        return np.vstack(inner)

    X_origin = X.copy()
    rng = np.random.default_rng()
    output = []
    output = Parallel(n_jobs=n_jobs,
                      max_nbytes=None)(delayed(predict)(model, X, rng, i)
                                       for i in progress_bar(range(X.shape[1]),
                                                             total=X.shape[1]))
    return np.array(output)


def kfold_predict_shuffle_inputs(model, X, y, times: int=100, n_jobs: int=1,
                                 cv: int=5):
    """Predict X by shuffling all the inputs using cross-validation"""

    if cv is None or isinstance(cv, int):
        n_splits = 5 if cv is None else cv
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    output = np.empty((X.shape[1], times, y.shape[0]))
    hy = np.empty_like(y)
    for tr, vs in cv.split(X, y):
        m = clone(model).fit(X[tr], y[tr])
        output[:, :, vs] = predict_shuffle_inputs(m, X[vs], times=times,
                                                  n_jobs=n_jobs)
        hy[vs] = m.predict(X[vs])
    return hy, output