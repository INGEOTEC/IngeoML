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
from IngeoML.cluster import farthest_first_traversal


def test_farthest_first_traversal():
    """Test"""
    data = np.array([[1, 0, 0, 0],
                     [1, 0, 1, 0],
                     [0, 1, 0, 0]])
    res = farthest_first_traversal(data, num=2)
    assert res == [0, 2]