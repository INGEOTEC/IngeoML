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
from typing import Callable
from itertools import product
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
import numpy as np
import jax
import jax.numpy as jnp
from jax import nn
import optax
from IngeoML.utils import Batches, balance_class_weights, progress_bar, soft_BER, cos_distance, cos_similarity


def optimize(parameters: object, batches: Batches,
             objective: Callable[[object, jnp.array, jnp.array], object],
             epochs: int=5, learning_rate: float=1e-2,
             every_k_schedule: int=None,
             n_iter_no_change: int=jnp.inf,
             validation=None,
             model: Callable[[object, jnp.array], jnp.array]=None,
             return_evolution: bool=None,
             validation_score=None,
             discretize_val: bool= True,
             optimizer=None,
             **kwargs):
    """Optimize
    
    :param parameters: Parameters to optimize.
    :param batches: Batches used in the optimization.
    :type batches: :py:class:`~IngeoML.utils.Batches`
    :param objective: Objective function.
    :param epochs: Number of epochs.
    :param learning_rate: Learning rate, default=1e-2.
    :type learning_rate: float
    :param every_k_schedule: Update the parameters every k, default=jnp.inf.
    :type every_k_schedule: int
    :param validation: Validation set.
    :param model: Model.
    :param return_evolution: Whether to use the evolution the validation scores, default=None.
    :type return_evolution: bool
    :param validation_score: Function to compute the validation-set performance.
    :param discretize_val: whether to transform one-hot encoding to labels
    :type discretize_val: True
    :param optimizer: Optimizer default optax.adam. 

    >>> import jax
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.preprocessing import OneHotEncoder
    >>> from IngeoML.utils import Batches
    >>> from IngeoML.utils import soft_BER
    >>> from IngeoML.optimizer import optimize
    >>> def model(params, X):
            Y = X @ params['W'] + params['W0']
            return Y
    >>> def objective(params, X, y, w):
            hy = model(params, X)
            hy = jax.nn.softmax(jnp.array(hy), axis=1)
            return soft_BER(y, hy)
    >>> model = jax.jit(model)
    >>> X, y = load_iris(return_X_y=True)
    >>> m = LinearSVC(dual='auto').fit(X, y)
    >>> parameters = dict(W=m.coef_.T,
                          W0=m.intercept_)
    >>> encoder = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    >>> y_enc = encoder.transform(y.reshape(-1, 1))
    >>> batches = Batches()
    >>> batches = [[jnp.array(X[idx]),
                    jnp.array(y_enc[idx]), None]
                   for idx in batches.split(y=y)]
    >>> optimize(parameters, batches, objective)
    {'W': Array([[ 0.18344977,  0.05524644, -0.8504886],
                 [ 0.4549369 , -0.9008946 , -0.9865761],
                 [-0.8149536 ,  0.409234  ,  1.3809077],
                 [-0.4335734 , -0.9606271 ,  1.8651136]], dtype=float32),
     'W0': Array([ 0.10852419,  1.6873716 , -1.710725], dtype=float32)}    
    """

    @jax.jit
    def update_finite(a, b):
        m = jnp.isfinite(b)
        return jnp.where(m, b, a)

    @jax.jit
    def evaluacion(parameters, estado, X, y, weights):
        grads = objective_grad(parameters, X, y, weights)
        updates, estado = optimizador.update(grads, estado, parameters)
        parameters = optax.apply_updates(parameters, updates)
        return parameters, estado

    def _validation_score():
        if validation is None:
            return - jnp.inf
        X, y = validation
        hy = model(parameters, X)
        if discretize_val:
            if y.ndim == 1:
                hy = np.where(hy.flatten() > 0, 1, 0)
            else:
                hy = hy.argmax(axis=1)
                y = y.argmax(axis=1)
        return validation_score(y, hy)

    def set_output(value):
        if return_evolution:
            return value, evolution
        return value

    if optimizer is None:
        optimizador = optax.adam(learning_rate=learning_rate, **kwargs)
    else:
        optimizador = optimize(learning_rate=learning_rate, **kwargs)
    if validation_score is None:
        validation_score = lambda y, hy: f1_score(y, hy, average='macro')
    total = epochs * len(batches)        
    if every_k_schedule is None or every_k_schedule > len(batches):
        every_k_schedule = len(batches)
    every_k_schedule = [x for x in range(every_k_schedule, len(batches) + 1)
                        if (total % x) == 0][0]
    optimizador = optax.MultiSteps(optimizador,
                                   every_k_schedule=every_k_schedule)
    estado = optimizador.init(parameters)
    objective_grad  = jax.grad(objective)
    fit = (1, _validation_score(), parameters)
    evolution = [fit[:2]]
    i = 1
    n_iter_no_change = n_iter_no_change * every_k_schedule
    for _, (X, y, weights) in progress_bar(product(range(epochs),
                                                   batches), total=total):
        p, estado = evaluacion(parameters, estado, X, y, weights)
        parameters = jax.tree_map(update_finite, parameters, p)
        if (i % every_k_schedule) == 0:
            comp = _validation_score()
            evolution.append((i, comp))
            if comp > fit[1]:
                fit = (i, comp, parameters)
            if comp >= 1:
                return set_output(fit[-1])
        if (i - fit[0]) > n_iter_no_change:
            return set_output(fit[-1])
        i += 1
    comp = _validation_score()
    evolution.append((i, comp))
    if validation is None or comp > fit[1]:
        return set_output(parameters)
    return set_output(fit[-1])


def estimator(parameters: object,
              model: Callable[[object, jnp.array], jnp.array],
              X, y,
              batches: Batches=None,
              array: Callable[[object],object]=jnp.array,
              class_weight: str='balanced',
              n_iter_no_change: int=jnp.inf,
              deviation=None, n_outputs: int=None, validation=None,
              discretize_val: bool= True,
              classifier: bool=True,
              **kwargs):
    """Estimator optimized with optax

    :param parameters: Parameters to optimize.
    :param model: Model.
    :param X: Independent variables.
    :param y: Dependent variable.
    :param batches: Batches used in the optimization.
    :type batches: :py:class:`~IngeoML.utils.Batches`
    :param array: Function to transform the independent variable.
    :param class_weight: Element weights.
    :param n_iter_no_change: Number of iterations without improving the performance.
    :type n_iter_no_change: int
    :param deviation: Deviation function between the actual and predicted values.
    :param n_output: Number of outputs.
    :param validation: Validation set, if None it is created with :py:class:`~sklearn.model_selection.ShuffleSplit` or :py:class:`~sklearn.model_selection.StratifiedShuffleSplit`
    :param discretize_val: whether to transform one-hot encoding to labels
    :type discretize_val: bool
    :param classifier: The estimator is classifier, default=True.
    :type classifier: bool

    >>> import jax
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import LinearSVC
    >>> from IngeoML.optimizer import estimator
    >>> def model(params, X):
            Y = X @ params['W'] + params['W0']
            return Y
    >>> model = jax.jit(model)
    >>> X, y = load_iris(return_X_y=True)
    >>> m = LinearSVC(dual='auto').fit(X, y)
    >>> parameters = dict(W=m.coef_.T,
                          W0=m.intercept_)
    >>> p, evolution = estimator(parameters, model, X, y,
                                 return_evolution=True)
    """

    @jax.jit
    def deviation_model_binary(params, X, y, weights):
        hy = model(params, X)
        hy = nn.sigmoid(hy)
        hy = hy.flatten()
        y_ = jnp.vstack((y, 1 - y)).T
        hy_ = jnp.vstack((hy, 1 - hy)).T        
        return deviation(y_, hy_, weights)

    @jax.jit
    def deviation_model(params, X, y, weights):
        hy = model(params, X)
        hy = nn.softmax(hy, axis=-1)
        return deviation(y, hy, weights)
    
    @jax.jit
    def deviation_regression(params, X, y, weights):
        hy = model(params, X)
        return deviation(y, hy, weights)

    def encode(y, n_outputs, validation):
        if n_outputs == 1:
            labels = np.unique(y)
            h = {v:k for k, v in enumerate(labels)}
            y_enc = np.array([h[x] for x in y])
            if validation is not None and not hasattr(validation, 'split'):
                _ = validation[1]
                validation[1] = np.array([h[x] for x in _])
        else:
            encoder = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
            y_enc = encoder.transform(y.reshape(-1, 1))
            if validation is not None and not hasattr(validation, 'split'):
                _ = validation[1]
                validation[1] = encoder.transform(_.reshape(-1, 1))
        return y_enc

    def create_batches(batches):
        if batches is None:
            batches = Batches(size=512 if X.shape[0] >= 2048 else 256,
                              random_state=0)
        batches_ = []
        if classifier and class_weight == 'balanced':
            splits = batches.split(y=y)
            balance = balance_class_weights
        else:
            splits = batches.split(X)
            balance = lambda x: jnp.ones(x.shape[0]) / x.shape[0]

        for idx in splits:
            batches_.append((array(X[idx]),
                             jnp.array(y_enc[idx]),
                             jnp.array(balance(y[idx]))))
        return batches_, splits

    def _validation(validation, X, y_enc, y):
        if validation is not None and hasattr(validation, 'split'):
            tr, vs = next(validation.split(X, y))
            validation = [array(X[vs]), jnp.array(y_enc[vs])]
            X, y_enc = X[tr], y_enc[tr]
            y = y[tr]
        elif validation is not None and not hasattr(validation, 'split'):
            validation = [array(validation[0]), jnp.array(validation[1])]
        return validation, X, y_enc, y

    def _objective(deviation):
        if not classifier:
            return deviation_regression, deviation
        if deviation is None:
            deviation = soft_BER
        if n_outputs == 1:
            objective = deviation_model_binary
        else:
            objective = deviation_model
        return objective, deviation

    if validation is None:
        cnt = X.shape[0]
        if cnt < 2048:
            test_size = 0.2
        else:
            test_size = 512
        if classifier and class_weight == 'balanced':
            validation = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        else:
            validation = ShuffleSplit(n_splits=1, test_size=test_size)
    if n_outputs is None:
        n_outputs = model(parameters, array(X[:1])).shape[-1]
    if classifier:
        y_enc = encode(y, n_outputs, validation)
    else:
        y_enc = y
    validation, X, y_enc, y = _validation(validation, X, y_enc, y)
    batches_, splits = create_batches(batches)
    if n_iter_no_change < jnp.inf and validation is None:
        jaccard = Batches.jaccard(splits)
        index = jaccard.argmin()
        validation = batches_[index][:2]
        del batches_[index]
    objective, deviation = _objective(deviation)
    return optimize(parameters, batches_, objective,
                    n_iter_no_change=n_iter_no_change,
                    validation=validation, model=model,
                    discretize_val=discretize_val,
                    **kwargs)


def classifier(parameters: object,
               model: Callable[[object, jnp.array], jnp.array],
               X, y,
               batches: Batches=None,
               array: Callable[[object],object]=jnp.array,
               class_weight: str='balanced',
               deviation=None, n_outputs: int=None, validation=None,
               discretize_val: bool= True,
               every_k_schedule=4,
               epochs=100, learning_rate=1e-4,
               n_iter_no_change=5,
               **kwargs):
    """Classifier optimized with optax

    :param parameters: Parameters to optimize.
    :param model: Model.
    :param X: Independent variables.
    :param y: Dependent variable.
    :param batches: Batches used in the optimization.
    :type batches: :py:class:`~IngeoML.utils.Batches`
    :param array: Function to transform the independent variable.
    :param class_weight: Element weights.
    :param deviation: Deviation function between the actual and predicted values.
    :param n_output: Number of outputs.
    :param validation: Validation set.
    :param discretize_val: whether to transform one-hot encoding to labels
    :type discretize_val: bool
    :param every_k_schedule: Update the parameters every k, default=4.
    :type every_k_schedule: int
    :param epochs: Number of epochs.
    :param learning_rate: Learning rate, default=1e-4.
    :type learning_rate: float
    :param n_iter_no_change: Number of iterations without improving the performance, default=5.
    :type n_iter_no_change: int


    >>> import jax
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import LinearSVC
    >>> from IngeoML.optimizer import classifier
    >>> def model(params, X):
            Y = X @ params['W'] + params['W0']
            return Y
    >>> model = jax.jit(model)
    >>> X, y = load_iris(return_X_y=True)
    >>> m = LinearSVC(dual='auto').fit(X, y)
    >>> parameters = dict(W=m.coef_.T,
                          W0=m.intercept_)
    >>> p, evolution = classifier(parameters, model, X, y,
                                  return_evolution=True)
    """

    return estimator(parameters, model, X, y, batches=batches,
                     array=array, class_weight=class_weight,
                     deviation=deviation, n_outputs=n_outputs,
                     validation=validation, discretize_val=discretize_val,
                     every_k_schedule=every_k_schedule,
                     epochs=epochs, learning_rate=learning_rate,
                     n_iter_no_change=n_iter_no_change,
                     **kwargs)


def regression(parameters: object,
               model: Callable[[object, jnp.array], jnp.array],
               X, y,
               deviation=cos_distance,
               discretize_val=False,
               classifier=False,
               validation_score=cos_similarity,
               every_k_schedule=4,
               epochs=100, learning_rate=1e-4,
               n_iter_no_change=5,
               **kwargs):
    """Regression optimized with optax

    :param parameters: Parameters to optimize.
    :param model: Model.
    :param X: Independent variables.
    :param y: Dependent variable.
    :param deviation: Deviation function between the actual and predicted values, default=cos_distance.
    :param discretize_val: whether to transform one-hot encoding to labels
    :type discretize_val: True
    :param classifier: The estimator is classifier, default=False.
    :type classifier: bool
    :param validation_score: Function to compute the validation-set performance.
    :param every_k_schedule: Update the parameters every k, default=4.
    :type every_k_schedule: int
    :param epochs: Number of epochs.
    :param learning_rate: Learning rate, default=1e-4.
    :type learning_rate: float
    :param n_iter_no_change: Number of iterations without improving the performance, default=5.
    :type n_iter_no_change: int

    >>> import jax
    >>> import jax.nn as nn
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import LinearRegression
    >>> from IngeoML.optimizer import regression
    >>> def model(params, X):
            Y = X @ params['W'] + params['W0']
            return nn.sigmoid(Y).flatten()
    >>> model = jax.jit(model)
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> m = LinearRegression().fit(X, y)
    >>> parameters = dict(W=m.coef_.T,
                          W0=m.intercept_)
    >>> p, evolution = regression(parameters, model, X, y,
                                  return_evolution=True)
    """    
    
    return estimator(parameters, model, X, y,
                     deviation=deviation,
                     discretize_val=discretize_val,
                     classifier=classifier,
                     validation_score=validation_score,
                     every_k_schedule=every_k_schedule,
                     epochs=epochs,
                     learning_rate=learning_rate,
                     n_iter_no_change=n_iter_no_change,
                     **kwargs) 