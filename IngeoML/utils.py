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

import jax
import jax.numpy as jnp
from jax import nn
import numpy as np
from sklearn.utils import check_random_state
try:
    USE_TQDM = True
    from tqdm import tqdm
except ImportError:
    USE_TQDM = False


def progress_bar(arg, **kwargs):
    """Progress bar using tqdm"""
    if USE_TQDM:
        return tqdm(arg, **kwargs)
    return arg


class Batches:
    """
    Helper class to create a set of batches.

    :param size: Bath size, default=64
    :type size: int
    :param strategy: Procedure to create the batch, default=stratified
    :type strategy: str
    :param remainder: Method used to deal with the remainder, default=fill
    :type remainder: str
    :param shuffle: Whether to shuffle the dataset, default=True
    :type shuffle: bool
    :param random_state: Random State, default=None

    >>> import numpy as np
    >>> from IngeoML.utils import Batches
    >>> b = Batches(size=3)
    >>> X = np.empty((5, 4))
    >>> b.split(X)
    array([[4, 0, 2],
           [1, 3, 4]])
    >>> y = np.r_[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2]
    >>> b.split(y=y)
    array([[ 0, 10,  5],
           [ 1,  6, 10],
           [ 2, 10,  7],
           [ 3, 10,  8],
           [10,  9,  4]])    
    """

    def __init__(self, size: int=64,
                 strategy: str='stratified',
                 remainder: str='fill',
                 shuffle: bool=True,
                 random_state: int=None) -> None:
        self.size = size
        self.strategy = strategy
        self.random_state = random_state
        self.remainder = remainder
        self.shuffle = shuffle

    @property
    def strategy(self):
        """Strategy"""
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        assert value in ['stratified', 'balanced']
        self._strategy = value

    @property
    def remainder(self):
        """Method to deal with the remainder"""
        return self._remainder

    @remainder.setter
    def remainder(self, value):
        assert value in ['fill', 'drop']
        self._remainder = value

    def _split_stratified(self, y: np.ndarray):
        dist = self.distribution(y, size=self.size)
        labels, cnt = np.unique(y, return_counts=True)
        if self.remainder == 'drop':
            rows = np.ceil(cnt / dist).min().astype(int)
        else:
            rows = np.ceil(y.shape[0] / self.size).astype(int)
        index = np.arange(y.shape[0])
        if self.shuffle:
            check_random_state(self.random_state).shuffle(index)
        output = []
        for label, columns in zip(labels, dist):
            mask = y == label
            output.append(self.blocks(index[mask], rows=rows,
                                      columns=columns))
        output = np.concatenate(output, axis=1)
        if self.shuffle:
            _ = [check_random_state(self.random_state).permutation(a)
                 for a in output]
            output = np.vstack(_)
        if self.remainder == 'drop' and np.any(rows * dist > cnt):
            return output[:-1]
        return output

    def _split_dataset(self, num_elements: int):
        index = np.arange(num_elements)
        fill = self.remainder == 'fill'
        if self.shuffle:
            check_random_state(self.random_state).shuffle(index)
        output = self.blocks(index, 
                             rows=np.ceil(index.shape[0] / self.size).astype(int),
                             columns=self.size)
        if index.shape[0] % self.size and not fill:
            return output[:-1]
        return output

    def blocks(self, index: np.ndarray,
               rows: int, columns: int):
        """Create the blocks
        :param index:
        :type index: np.ndarray
        :param rows: Number of rows
        :type rows: int
        :param columns: Number of columns
        :type columns: int

        >>> from IngeoML.utils import Batches
        >>> b = Batches(size=3)
        >>> b.blocks(np.arange(3), columns=2, rows=3)
        array([[0, 1],
               [2, 0],
               [1, 2]])        
        """
        num_elements = rows * columns
        if index.shape[0] < num_elements:
            cnt = np.ceil(rows * columns / index.shape[0])
            index = np.tile(index, cnt.astype(int))
            frst = index[:num_elements]
            rest = index[num_elements:].copy()
            check_random_state(self.random_state).shuffle(rest)
        elif index.shape[0] > num_elements:
            frst = index[:num_elements]
            rest = None
        else:
            frst = index
            rest = None
        if rest is not None:
            frst = np.concatenate((frst, rest))[:num_elements]
        frst.shape = (rows, columns)
        return frst

    @staticmethod
    def distribution(y: np.ndarray, size: int=64):
        """Distribution

        :param y: Labels
        :type y: np.ndarray
        :param size: Size of the batch
        :type param: int
        """

        _, cnt = np.unique(y, return_counts=True)
        dist = np.round(size * cnt / cnt.sum()).astype(int)
        missing = dist == 0
        dist[missing] = 1
        inc = size - dist.sum()
        if inc < 0:
            for _ in range(inc, 0):
                avail = np.where(dist > 1)[0]
                index = np.random.randint(0, avail.shape[0])
                dist[index] -= 1
        return dist

    def split(self, D=None, y: np.ndarray=None)->np.ndarray:
        """Method to create the batches

        :param D: Dataset
        :param y: Labels
        :type y: np.ndarray
        """
        if y is None:
            return self._split_dataset(D.shape[0])
        if self.strategy == 'stratified':
            return self._split_stratified(y)
        raise NotImplementedError(f'Missing {self.strategy}')

    @staticmethod
    def jaccard(splits: np.ndarray) -> np.ndarray:
        """Jaccard index between splits"""
        num_elem = np.unique(splits).shape[0]
        mask = np.empty(splits.shape[0], dtype=bool)
        output = np.empty(splits.shape[0])
        for i in range(splits.shape[0]):
            mask.fill(True)
            mask[i] = False
            rest = splits[mask].flatten()
            origin = splits[i]
            _ = np.intersect1d(origin, rest)
            output[i] = _.shape[0] / num_elem
        return output


def balance_class_weights(labels) -> np.ndarray:
    """Weights of the labels set to balance
    
    >>> import numpy as np
    >>> from IngeoML.utils import balance_class_weights
    >>> balance_class_weights(np.array(['a', 'a', 'b']))
    array([0.25, 0.25, 0.5 ])
    """

    y_ = labels
    labels, cnts = np.unique(y_, return_counts=True)
    weights = np.empty(y_.shape[0])
    for label, cnt in zip(labels, cnts):
        mask = y_ == label
        weights[mask] = 1 / (labels.shape[0] * cnt)
    return weights


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