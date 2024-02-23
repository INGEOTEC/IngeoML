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
import numpy as np
from IngeoML.utils import progress_bar


def feature_importance(model, X, y, predictions,
                       score=None):
    """Estimate the feature importance of the model"""
    if score is None:
        score = lambda y, hy: f1_score(y, hy, average='macro')
    base = score(y, model.predict(X))
    hy = np.array([[score(y, j) for j in i]
                  for i in progress_bar(predictions)])
    return base - hy


def predict_shuffle_inputs(model, X, times: int=100):
    """Predict X by shuffling all the inputs"""
    X_origin = X.copy()
    rng = np.random.default_rng()
    output = []
    for i in progress_bar(range(X.shape[1]), total=X.shape[1]):
        inner = []
        for _ in range(times):
            rng.shuffle(X[:, i])
            inner.append(model.predict(X))
        X = X_origin.copy()
        output.append(np.vstack(inner))
    return np.array(output)


