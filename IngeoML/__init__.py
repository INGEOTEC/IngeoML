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
import logging
from IngeoML.feature_selection import SelectFromModelCV
from IngeoML.analysis import feature_importance
from IngeoML.analysis import predict_shuffle_inputs, kfold_predict_shuffle_inputs
from IngeoML.supervised_learning import ConvexClassifier
logger = logging.getLogger(__name__)

try:
    from IngeoML.optimizer import classifier, regression
except ImportError:
    logger.warning('Install jax and optax to use IngeoML.optimizer')

__version__ = '0.0.28'
