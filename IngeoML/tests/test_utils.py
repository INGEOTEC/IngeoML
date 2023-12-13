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
from IngeoML.utils import Batches, balance_class_weigths, cross_entropy, error


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
                    [0.6, 0.4],
                    [0.6, 0.4]])
    w = jnp.array([1/3, 1/3, 1/3])
    value = error(y, hy, w)
    # assert value is None
    assert jnp.fabs(value - 0.33333334) < 1e-6