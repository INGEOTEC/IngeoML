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


def farthest_first_traversal(X: np.ndarray, num: int=512):
    """
    :param X: Elements 
    :type X: np.ndarray
    :param num: Number of elements
    :type num: int

    >>> from IngeoML.cluster import farthest_first_traversal
    >>> import numpy as np
    >>> data = np.array([[1, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0]])
    >>> farthest_first_traversal(data, num=2)
    [0, 2]
    """
    S = []
    siguiente = 0
    mask = np.ones(X.shape[0], dtype=bool)
    for i in range(1, num):
        mask[siguiente] = False
        dis = np.fabs(np.dot(X, X[siguiente]))
        S.append((siguiente, dis))
        if len(S) == 1:
            siguiente = S[0][1].argmin()
        else:
            _ = np.vstack([x for _, x in S]).max(axis=0)
            index = np.where(mask)[0]
            siguiente = index[_[index].argmin()]
    index = [i for i, _ in S]
    index.append(siguiente)
    return sorted(index)