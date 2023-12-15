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
import jax.numpy as jnp
from jax import nn
import jax
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder
from IngeoML.utils import Batches, balance_class_weigths, cross_entropy, error, error_binary


def test_batches():
    """Test Batches"""

    b = Batches(size=3)
    X = np.empty((5, 4))
    idx = b.split(X)
    assert idx.shape[0] == 2
    b.remainder = 'drop'
    idx2 = b.split(X)
    assert idx2.shape[0] == 1


def test_distribution():
    """Distribution"""

    y = np.r_[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2]
    dist = Batches.distribution(y, size=5)
    assert np.all(dist == np.array([2, 2, 1]))


def test_stratified():
    """Stratified batches"""
    y = np.r_[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2]
    batch = Batches(size=5, shuffle=False)
    output = batch.split(y=y)
    assert np.all(output[:, -1] == 10)
    batch.shuffle =True
    batch.split(y=y)
    

def test_balance_class_weigths():
    """Weights to have a balance in the labels"""
    y = np.r_[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2]
    w = balance_class_weigths(y)
    assert w.sum() == 1
    assert w.shape[0] == y.shape[0]


def test_batches_nofill():
    """Test stratified no fill"""

    batches = Batches(size=4,
                      shuffle=False,
                      remainder='drop')
    y = np.r_[0, 0, 0, 0, 0, 0,
              1, 1, 1, 1, 1, 2]
    res = batches.split(y=y)
    assert res.shape[0] == 1
    y = np.r_[0, 0, 0, 0, 0, 0,
              1, 1, 1, 1, 1]
    res = batches.split(y=y)    
    _, b = np.unique(res, return_counts=True)
    assert np.all(b <= 1)
    assert res.shape[0] == 2


def test_batches_jaccard():
    """Test jaccard index"""
    batches = Batches(size=4,
                      shuffle=False)
    y = np.r_[0, 0, 0, 0, 0, 0,
              1, 1, 1, 1]
    splits = batches.split(y=y)
    res = batches.jaccard(splits)
    assert res.shape[0] == splits.shape[0]
    assert res[0] == 0.2


def test_cross_entropy():
    y = jnp.array([[1, 0],
                   [1, 0],
                   [0, 1]])
    hy = jnp.array([[0.9, 0.1],
                    [0.6, 0.4],
                    [0.2, 0.8]])
    w = jnp.array([1/3, 1/3, 1/3])
    value = cross_entropy(y, hy, w)
    assert value == 0.27977654
    hy = jnp.array([[1, 0],
                    [1, 0],
                    [0.01, 0.99]])
    value = cross_entropy(y, hy, w)
    assert jnp.fabs(value - 0.00335011) < 1e-6
    value = cross_entropy(y, y, w)
    assert value == 0
    y = jnp.array([1, 0, 1])
    hy = jnp.array([0.9, 0.3, 0.8])
    w = jnp.array([1/3, 1/3, 1/3])
    value = cross_entropy(y, hy, w)
    assert jnp.fabs(value - 0.3285041) < 1e-6


def test_error():
    y = jnp.array([[1, 0],
                   [1, 0],
                   [0, 1]])
    hy = jnp.array([[0.9, 0.1],
                    [0.49, 1 - 0.49],
                    [0.1, 0.9]])
    w = jnp.array([1/3, 1/3, 1/3])
    value = error(y, hy, w)
    # assert value is None
    assert jnp.fabs(value - 0.33331817) < 1e-6


def test_error_binary():
    y = jnp.array([1, 0, 1])
    hy = jnp.array([1, 0.55, 1])
    w = jnp.array([1/3, 1/3, 1/3])
    value = error_binary(y, hy, w)
    # assert value is None
    assert jnp.fabs(value - 0.3333333) < 1e-6


def test_error_grad():
    @jax.jit
    def modelo(params, X):
        Y = X @ params['W'] + params['W0']
        return Y
    
    @jax.jit
    def deviation_model(params, X, y, weigths):
        hy = modelo(params, X)
        hy = jax.nn.softmax(hy, axis=-1)
        return error(y, hy, weigths)        
    
    X, y = load_iris(return_X_y=True)
    encoder = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    y_enc = encoder.transform(y.reshape(-1, 1))
    m = LinearSVC(dual='auto').fit(X, y)
    parameters = dict(W=m.coef_.T,
                      W0=m.intercept_)
    grad = jax.grad(deviation_model)
    w = jnp.ones(y.shape[0]) / y.shape[0]
    p = grad(parameters, X, y_enc, w)
    # assert p is None
    assert jnp.fabs(p['W']).sum() > 0


def test_error_binary_grad():
    @jax.jit
    def modelo(params, X):
        Y = X @ params['W'] + params['W0']
        return Y
    
    @jax.jit
    def deviation_model(params, X, y, weigths):
        hy = modelo(params, X)
        hy = nn.sigmoid(hy)
        hy = hy.flatten()        
        return error_binary(y, hy, weigths)        
    
    X, y = load_breast_cancer(return_X_y=True)
    labels = np.unique(y)            
    h = {v:k for k, v in enumerate(labels)}
    y_enc = np.array([h[x] for x in y])
    m = LinearSVC(dual='auto').fit(X, y)
    parameters = dict(W=m.coef_.T,
                      W0=m.intercept_)
    grad = jax.grad(deviation_model)
    w = jnp.ones(y.shape[0]) / y.shape[0]
    p = grad(parameters, X, y_enc, w)
    # assert p is None
    assert jnp.fabs(p['W']).sum() > 0    