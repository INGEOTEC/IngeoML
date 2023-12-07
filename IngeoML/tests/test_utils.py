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
from IngeoML.utils import Batches


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
    output2 = batch.split(y=y)
    