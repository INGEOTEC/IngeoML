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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from IngeoML.utils import Batches, balance_class_weights, cross_entropy, soft_error, soft_recall, soft_BER, soft_precision, soft_f1_score, soft_comp_macro_f1, cos_distance, pearson, pearson_distance, pearson_similarity


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
    

def test_balance_class_weights():
    """Weights to have a balance in the labels"""
    y = np.r_[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2]
    w = balance_class_weights(y)
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


def test_soft_error():
    """Test soft error"""
    y = jnp.array([[1, 0],
                   [1, 0],
                   [0, 1]])
    hy = jnp.array([[0.9, 0.1],
                    [0.49, 1 - 0.49],
                    [0.1, 0.9]])
    w = jnp.array([1/3, 1/3, 1/3])
    value = soft_error(y, hy, w)
    # assert value is None
    assert jnp.fabs(value - 0.33331817) < 1e-6


def test_soft_error_grad():
    """Test soft error grad"""
    @jax.jit
    def modelo(params, X):
        Y = X @ params['W'] + params['W0']
        return Y
    
    @jax.jit
    def deviation_model(params, X, y, weights):
        hy = modelo(params, X)
        hy = jax.nn.softmax(hy, axis=-1)
        return soft_error(y, hy, weights)        
    
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


def test_soft_recall():
    """test soft recall"""
    y = jnp.array([[1, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1]])
    hy = jnp.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0]])
    res = soft_recall(y, hy)
    assert np.all(res == jnp.array([0.5, 1, 0.25]))


def test_soft_BER():
    """Test soft Balanced Error Rate (BER)"""
    y = jnp.array([[1, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1]])
    hy = jnp.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0]])
    res = soft_BER(y, hy)
    assert jnp.fabs(res - 0.41666666) < 1e-6
    w = balance_class_weights(y.argmax(axis=1))
    r = soft_error(y, hy, w)
    assert np.fabs(r - res) < 1e-6


def test_soft_BER_grad():
    """Test soft-BER grad"""

    @jax.jit
    def modelo(params, X):
        Y = X @ params['W'] + params['W0']
        return Y
    
    @jax.jit
    def deviation_model(params, X, y, weights=None):
        hy = modelo(params, X)
        hy = jax.nn.softmax(hy, axis=-1)
        return soft_BER(y, hy, weights=weights)
    
    X, y = load_iris(return_X_y=True)
    encoder = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    y_enc = encoder.transform(y.reshape(-1, 1))
    m = LinearSVC(dual='auto').fit(X, y)
    parameters = dict(W=m.coef_.T,
                      W0=m.intercept_)
    grad = jax.grad(deviation_model)
    p = grad(parameters, X, y_enc)
    assert jnp.fabs(p['W']).sum() > 0


def test_soft_precision():
    """Test soft precision"""
    y = jnp.array([[1, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1]])
    hy = jnp.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0]])
    res = soft_precision(y, hy)
    assert np.all(res == jnp.array([0.25, 0.5, 1]))


def test_soft_f1_score():
    """Test soft precision"""
    y = jnp.array([[1, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1]])
    hy = jnp.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0]])
    res = soft_f1_score(y, hy)
    diff = np.fabs(res - jnp.array([0.33333334, 0.6666667, 0.4]))    
    assert np.all(diff < 1e-6)


def test_soft_comp_macro_f1_grad():
    """Test soft-BER grad"""

    @jax.jit
    def modelo(params, X):
        Y = X @ params['W'] + params['W0']
        return Y

    @jax.jit
    def deviation_model(params, X, y, weights=None):
        hy = modelo(params, X)
        hy = jax.nn.softmax(hy, axis=-1)
        return soft_comp_macro_f1(y, hy)

    X, y = load_iris(return_X_y=True)
    encoder = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    y_enc = encoder.transform(y.reshape(-1, 1))
    m = LinearSVC(dual='auto').fit(X, y)
    parameters = dict(W=m.coef_.T,
                      W0=m.intercept_)
    grad = jax.grad(deviation_model)
    p = grad(parameters, X, y_enc)
    assert jnp.fabs(p['W']).sum() > 0


def test_cos_distance():
    """Test cos distance"""

    y = jnp.array([1, 0, 1])
    hy = jnp.array([0.9, 0, 0.8])
    dis = cos_distance(y, hy, None)
    hy = jnp.array([1, 0, 1])
    dis_c = cos_distance(y, hy, None)
    assert dis > dis_c
    assert np.all(dis_c < 1e-6)
    hy = jnp.array([0, 1, 0])
    dis = cos_distance(y, hy, None)
    assert np.all(dis - 1 < 1e-6)


def test_cos_distance_grad():
    """Test cos_distance grad"""

    @jax.jit
    def modelo(params, X):
        Y = X @ params['W'] + params['W0']
        return nn.sigmoid(Y).flatten()
    
    def objective(params, X, y):
        hy = modelo(params, X)
        return cos_distance(y, hy, None)
    
    X, y = load_breast_cancer(return_X_y=True)
    m = LinearRegression().fit(X, y)
    params = dict(W=jnp.array(m.coef_.T),
                  W0=jnp.array(m.intercept_))
    grad = jax.grad(objective)
    p = grad(params, X, y)
    assert jnp.fabs(p['W']).sum() > 0


def test_pearson():
    """Test cos distance"""

    from scipy.stats import pearsonr
    y = jnp.array([1, 0, 1])
    hy = jnp.array([0.9, 0, 0.8])

    dis = pearson(y, hy, None)
    value = pearsonr(y, hy).statistic
    diff = value - dis
    assert jnp.fabs(diff) < 1e-7
    value2 = pearson_similarity(y, hy)
    assert value2 > value 
    # hy = jnp.array([1, 0, 1])
    # dis_c = cos_distance(y, hy, None)
    # assert dis > dis_c
    # assert np.all(dis_c < 1e-6)
    # hy = jnp.array([0, 1, 0])
    # dis = cos_distance(y, hy, None)
    # assert np.all(dis - 1 < 1e-6)


def test_pearson_distance_grad():
    """Test pearson_distance grad"""

    @jax.jit
    def modelo(params, X):
        Y = X @ params['W'] + params['W0']
        return nn.sigmoid(Y).flatten()
    
    def objective(params, X, y):
        hy = modelo(params, X)
        return pearson_distance(y, hy, None)
    
    X, y = load_breast_cancer(return_X_y=True)
    m = LinearRegression().fit(X, y)
    params = dict(W=jnp.array(m.coef_.T),
                  W0=jnp.array(m.intercept_))
    grad = jax.grad(objective)
    p = grad(params, X, y)
    assert jnp.fabs(p['W']).sum() > 0