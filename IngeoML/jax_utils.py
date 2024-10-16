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
import jax
import jax.numpy as jnp
from jax import nn


@jax.jit
def cross_entropy(y: jnp.array, hy: jnp.array, weights: jnp.array) -> jnp.array:
    """Cross-entropy loss
    
    :param y: Gold standard
    :param hy: Predictions
    :param weights: Weights for each element

    >>> import jax.numpy as jnp
    >>> from IngeoML.utils import cross_entropy
    >>> y = jnp.array([[1, 0],
                       [1, 0],
                       [0, 1]])
    >>> hy = jnp.array([[0.9, 0.1],
                        [0.6, 0.4],
                        [0.2, 0.8]])
    >>> w = jnp.array([1/3, 1/3, 1/3])
    >>> cross_entropy(y, hy, w)
    Array(0.27977654, dtype=float32)    
    """

    values = - ((y * jnp.log(hy)).sum(axis=-1) * weights)
    return jnp.nansum(values)


@jax.jit
def soft_error(y: jnp.array, hy: jnp.array, weights: jnp.array) -> jnp.array:
    """Soft Error

    :param y: Gold standard
    :param hy: Predictions
    :param weights: Weights for each element

    >>> import jax.numpy as jnp
    >>> from IngeoML.utils import soft_error
    >>> y = jnp.array([[1, 0],
                       [1, 0],
                       [0, 1]])
    >>> hy = jnp.array([[0.9, 0.1],
                        [0.4999, 1 - 0.4999],
                        [0.1, 0.9]])
    >>> w = jnp.array([1/3, 1/3, 1/3])
    >>> soft_error(y, hy, w)
    Array(0.17499197, dtype=float32)    
    """

    res = y * hy
    res = res.sum(axis=-1) - 1 / y.shape[1]
    return 1 - (nn.sigmoid(1e3 * res) * weights).sum(axis=-1)


@jax.jit
def soft_BER(y: jnp.array, hy: jnp.array, weights=None) -> jnp.array:
    """Soft Balanced Error Rate
    
    :param y: Gold standard
    :param hy: Predictions
    :param weights: Weights are not used

    >>> import jax.numpy as jnp
    >>> from IngeoML.utils import soft_BER
    >>> y = jnp.array([[1, 0],
                       [1, 0],
                       [0, 1]])
    >>> hy = jnp.array([[0.9, 0.1],
                        [0.4999, 1 - 0.4999],
                        [0.1, 0.9]])
    >>> w = jnp.array([1/3, 1/3, 1/3])
    >>> soft_BER(y, hy, w)
    Array(0.13124394, dtype=float32)        
    """

    return 1 - soft_recall(y, hy).mean()


@jax.jit
def soft_recall(y: jnp.array, hy: jnp.array, weights=None) -> jnp.array:
    """Soft Recall

    :param y: Gold standard
    :param hy: Predictions
    :param weights: Weights are not used

    >>> import jax.numpy as jnp
    >>> from IngeoML.utils import soft_recall
    >>> y = jnp.array([[1, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 1],
                       [0, 0, 1],
                       [0, 0, 1]])
    >>> hy = jnp.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0]])
    >>> soft_recall(y, hy)
    Array([0.5, 1.0, 0.25], dtype=float32)
    """

    hy = nn.sigmoid((hy - 1 / y.shape[1]) * 1e3)
    res = y * hy
    return res.sum(axis=0) / y.sum(axis=0)


@jax.jit
def soft_precision(y: jnp.array, hy: jnp.array, weights=None) -> jnp.array:
    """Soft Recall

    :param y: Gold standard
    :param hy: Predictions
    :param weights: Weights are not used

    >>> import jax.numpy as jnp
    >>> from IngeoML.utils import soft_precision
    >>> y = jnp.array([[1, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 1],
                       [0, 0, 1],
                       [0, 0, 1]])
    >>> hy = jnp.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0]])
    >>> soft_precision(y, hy)
    Array([0.33333334, 0.33333334, 1.0], dtype=float32)
    """

    hy = nn.sigmoid((hy - 1 / y.shape[1]) * 1e3)
    res = y * hy
    return res.sum(axis=0) / hy.sum(axis=0)


@jax.jit
def soft_f1_score(y: jnp.array, hy: jnp.array, weights=None) -> jnp.array:
    """Soft F1 score

    :param y: Gold standard
    :param hy: Predictions
    :param weights: Weights are not used

    >>> import jax.numpy as jnp
    >>> from IngeoML.utils import soft_f1_score
    >>> y = jnp.array([[1, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 1],
                       [0, 0, 1],
                       [0, 0, 1]])
    >>> hy = jnp.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0]])
    >>> soft_f1_score(y, hy)
    Array([0.4, 0.5, 0.4], dtype=float32)
    """

    recall = soft_recall(y, hy)
    precision = soft_precision(y, hy)
    return 2 * recall * precision / (recall + precision)


@jax.jit
def soft_comp_macro_f1(y: jnp.array, hy: jnp.array, weights=None) -> jnp.array:
    """Soft Complement macro-F1

    :param y: Gold standard
    :param hy: Predictions
    :param weights: Weights are not used

    >>> import jax.numpy as jnp
    >>> from IngeoML.utils import soft_comp_macro_f1
    >>> y = jnp.array([[1, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 1],
                       [0, 0, 1],
                       [0, 0, 1]])
    >>> hy = jnp.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0]])
    >>> soft_comp_macro_f1(y, hy)
    Array(0.56666666, dtype=float32)    
    """

    return 1 - soft_f1_score(y, hy).mean()


@jax.jit
def soft_comp_weighted_f1(y: jnp.array, hy: jnp.array, weights: jnp.array) -> jnp.array:
    """Soft Complement weighted-F1

    :param y: Gold standard
    :param hy: Predictions
    :param weights: Weights are not used

    >>> import jax.numpy as jnp
    >>> from IngeoML.utils import soft_comp_weighted_f1
    >>> y = jnp.array([[1, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 1],
                       [0, 0, 1],
                       [0, 0, 1]])
    >>> hy = jnp.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0]])
    >>> soft_comp_weighted_f1(y, hy)
    Array(0.5857142, dtype=float32)    
    """

    return 1 - (soft_f1_score(y, hy) * weights).sum()

@jax.jit
def cos_similarity(y: jnp.array, hy: jnp.array, weights=None) -> jnp.array:
    """Cos similarity

    :param y: Gold standard
    :param hy: Predictions
    :param weights: Weights are not used

    >>> import jax.numpy as jnp
    >>> from IngeoML.utils import cos_similarity
    >>> y = jnp.array([1, 0, 1])
    >>> hy = jnp.array([0, 1, 0])
    >>> cos_similarity(y, hy, None)
    Array(0., dtype=float32)    
    """
    y = y / jnp.linalg.norm(y)
    hy = hy / jnp.linalg.norm(hy)
    return jnp.dot(y, hy)


@jax.jit
def cos_distance(y: jnp.array, hy: jnp.array, weights=None) -> jnp.array:
    """Cos distance

    :param y: Gold standard
    :param hy: Predictions
    :param weights: Weights are not used

    >>> import jax.numpy as jnp
    >>> from IngeoML.utils import cos_distance
    >>> y = jnp.array([1, 0, 1])
    >>> hy = jnp.array([0, 1, 0])
    >>> cos_distance(y, hy, None)
    Array(1., dtype=float32)    
    """
    return 1 - jnp.fabs(cos_similarity(y, hy))


@jax.jit
def pearson(y: jnp.array, hy: jnp.array, weights=None) -> jnp.array:
    """Pearson correlation

    :param y: Gold standard
    :param hy: Predictions
    :param weights: Weights are not used

    >>> import jax.numpy as jnp
    >>> from IngeoML.utils import pearson
    >>> y = jnp.array([1, 0, 1])
    >>> hy = jnp.array([0.9, 0.1, 0.8])
    >>> pearson(y, hy, None)
    Array(0.9933992, dtype=float32)    
    """
    
    mu_y = y.mean()
    mu_hy = hy.mean()
    frst = (y - mu_y)
    scnd = (hy - mu_hy)
    num = (frst * scnd).sum()
    den = jnp.sqrt((frst**2).sum()) * jnp.sqrt((scnd**2).sum())
    return num / den


@jax.jit
def pearson_distance(y: jnp.array, hy: jnp.array, weights=None) -> jnp.array:
    """Pearson correlation

    :param y: Gold standard
    :param hy: Predictions
    :param weights: Weights are not used

    >>> import jax.numpy as jnp
    >>> from IngeoML.utils import pearson_distance
    >>> y = jnp.array([1, 0, 1])
    >>> hy = jnp.array([0.9, 0.1, 0.8])
    >>> pearson_distance(y, hy, None)
    Array(0.0033004, dtype=float32)
    """
    
    value = pearson(y, hy)
    return -(value - 1) / 2


@jax.jit
def pearson_similarity(y: jnp.array, hy: jnp.array, weights=None) -> jnp.array:
    """Pearson correlation

    :param y: Gold standard
    :param hy: Predictions
    :param weights: Weights are not used

    >>> import jax.numpy as jnp
    >>> from IngeoML.utils import pearson_distance
    >>> y = jnp.array([1, 0, 1])
    >>> hy = jnp.array([0.9, 0.1, 0.8])
    >>> pearson_distance(y, hy, None)
    Array(0.0033004, dtype=float32)
    """
    
    value = pearson(y, hy) + 1
    return value / 2