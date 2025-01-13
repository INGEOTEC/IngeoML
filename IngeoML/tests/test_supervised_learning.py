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
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.naive_bayes import GaussianNB
from IngeoML.supervised_learning import ConvexClassifier


def test_ConvexClassifier_2cl():
    """test ConvexClassifier"""
    X, y = load_breast_cancer(return_X_y=True)
    uno = make_pipeline(Normalizer(),
                        LinearSVC(class_weight='balanced'))
    dos = make_pipeline(Normalizer(),
                        LogisticRegression(class_weight='balanced'))
    tres = QDA()
    cuatro = GaussianNB()
    cinco = RandomForestClassifier()

    convex = ConvexClassifier()
    stack = StackingClassifier([('uno', uno),
                                ('dos', dos),
                                ('tres', tres),
                                ('cuatro', cuatro),
                                ('cinco', cinco)],
                                final_estimator=convex).fit(X, y)
    assert stack.final_estimator_.mixer[2] > 0.45
    hy = stack.predict_proba(X)
    assert hy.shape == (X.shape[0], 2)
    hy = stack.predict(X)
    assert (y == hy).mean() > 0.95


def test_ConvexClassifier_kcl():
    """test ConvexClassifier"""
    X, y = load_iris(return_X_y=True)
    uno = make_pipeline(Normalizer(),
                        LinearSVC(class_weight='balanced'))
    dos = make_pipeline(Normalizer(),
                        LogisticRegression(class_weight='balanced'))
    tres = QDA()
    cuatro = GaussianNB()
    cinco = RandomForestClassifier()

    convex = ConvexClassifier()
    stack = StackingClassifier([('uno', uno),
                                ('dos', dos),
                                ('tres', tres),
                                ('cuatro', cuatro),
                                ('cinco', cinco)],
                                final_estimator=convex).fit(X, y)
    # assert stack.final_estimator_ is None
    assert stack.final_estimator_.classes.shape[0] == 3
    assert stack.final_estimator_.mixer.shape[0] == 5
    stack.predict(X)


def test_ConvexClassifier_mixer():
    """Test negative values in convex combination"""
    convex = ConvexClassifier()
    convex.mixer = np.array([-0.01, 0.8, 0.2])
    assert np.all(convex.mixer == np.array([0, 0.8, 0.2]))