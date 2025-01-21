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

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.datasets import load_wine, load_digits
from IngeoML.feature_selection import SelectFromModelCV
from IngeoML.feature_selection import SelectFromLinearSVC


def test_SelectFromModelCV():
    X, y = load_wine(return_X_y=True)
    select = SelectFromModelCV(estimator=LinearSVC(dual='auto'),
                               scoring=lambda y, hy: f1_score(y, hy, average='macro'),
                               prefit=False)
    select.fit(X, y)
    ss = max([(k, v) for k, v in select.cv_results_.items()],
             key=lambda x: x[1])
    sum_support = (select.get_support()).sum()
    assert ss[0] == sum_support
    assert select.transform(X).shape[1] == sum_support


def test_SelectFromModelCV_prefit():
    X, y = load_wine(return_X_y=True)
    select = SelectFromModelCV(estimator=LinearSVC(dual='auto').fit(X, y),
                               scoring=lambda y, hy: f1_score(y, hy, average='macro'),
                               prefit=True)
    select.fit(X, y)
    ss = sorted([(k, v) for k, v in select.cv_results_.items()], key=lambda x: x[1])
    sum_support = (select.get_support()).sum()
    assert ss[-1][0] == sum_support
    assert select.transform(X).shape[1] == sum_support


def test_SelectFromLinearSVC():
    select = SelectFromLinearSVC()
    cl = LinearSVC(class_weight='balanced')
    X, y = load_digits(return_X_y=True)
    pipe = make_pipeline(select, cl).fit(X, y)
    hy = pipe.predict(X)
    assert hy.shape[0] == X.shape[0]
    assert pipe.steps[0][1].features.sum() < X.shape[1]
    