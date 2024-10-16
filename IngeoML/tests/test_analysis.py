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
from sklearn.datasets import load_iris, load_digits
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import LinearSVC
from IngeoML.analysis import feature_importance, predict_shuffle_inputs, kfold_predict_shuffle_inputs


def test_feature_importance():
    """Test feature importance"""

    X, y = load_iris(return_X_y=True)
    split = ShuffleSplit(n_splits=1, train_size=0.7).split(X, y)
    tr, vs = next(split)
    m = LinearSVC(dual='auto').fit(X[tr], y[tr])
    predictions = predict_shuffle_inputs(m, X[vs], times=97)
    diff = feature_importance(y[vs], m.predict(X[vs]), predictions)
    assert diff.shape == (4, 97)


def test_predict_shuffle_inputs():
    """Test predict_shuffle_inputs"""

    X, y = load_iris(return_X_y=True)
    split = ShuffleSplit(n_splits=1, train_size=0.7).split(X, y)
    tr, vs = next(split)
    m = LinearSVC(dual='auto').fit(X[tr], y[tr])
    hy = predict_shuffle_inputs(m, X[vs], n_jobs=-1)
    assert hy.shape == (4, 100, vs.shape[0])


def test_kfold_predict_shuffle_inputs():
    """Test kfold_predict_shuffle_inputs"""

    X, y = load_iris(return_X_y=True)
    m = LinearSVC(dual='auto')
    hy, predictions = kfold_predict_shuffle_inputs(m, X, y)
    assert predictions.shape == (4, 100, 150)
    assert hy.shape == (150, )
    diffs = feature_importance(y, hy, predictions)
    assert diffs.shape == (4, 100)