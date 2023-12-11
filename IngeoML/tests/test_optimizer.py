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
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import jax.numpy as jnp
import jax
from IngeoML.optimizer import adam, classifier
from IngeoML.utils import Batches


def test_adam():
    @jax.jit
    def modelo(params, X):
        Y = X @ params['W'] + params['W0']
        return Y
    
    @jax.jit
    def media_entropia_cruzada(params, X, y, pesos):
        hy = modelo(params, X)
        hy = jax.nn.softmax(jnp.array(hy), axis=0)
        return - ((y * jnp.log(hy)).sum(axis=1) * pesos).sum()    

    X, y = load_iris(return_X_y=True)
    m = LinearSVC(dual='auto').fit(X, y)
    parameters = dict(W=m.coef_.T,
                      W0=m.intercept_)
    encoder = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    batches = Batches()
    a = jnp.array
    y_enc = encoder.transform(y.reshape(-1, 1))
    batches = [[a(X[idx]), a(y_enc[idx])]
               for idx in batches.split(y=y)]
    pesos = jnp.ones(batches[0][0].shape[0])
    for b in batches:
        b.append(pesos)
    p = adam(parameters, batches, media_entropia_cruzada)
    assert np.fabs(p['W'] - parameters['W']).sum() > 0
    fit1 = media_entropia_cruzada(parameters, *batches[0])
    fit2 = media_entropia_cruzada(p, *batches[0])    
    assert fit2 < fit1


def test_classifier():
    """Classifier optimize with jax"""
    @jax.jit
    def modelo(params, X):
        Y = X @ params['W'] + params['W0']
        return Y

    X, y = load_iris(return_X_y=True)
    m = LinearSVC(dual='auto').fit(X, y)
    parameters = dict(W=jnp.array(m.coef_.T),
                      W0=jnp.array(m.intercept_))
    p = classifier(parameters, modelo, X, y)
    assert np.fabs(p['W'] - parameters['W']).sum() > 0
    diff = p['W0'] - parameters['W0']
    assert np.fabs(diff).sum() > 0
    X, y = load_breast_cancer(return_X_y=True)
    m = LinearSVC(dual='auto').fit(X, y)
    parameters = dict(W=jnp.array(m.coef_.T),
                      W0=jnp.array(m.intercept_))
    p2 = classifier(parameters, modelo, X, y)
    diff = p2['W0'] - parameters['W0']
    assert np.fabs(diff).sum() > 0