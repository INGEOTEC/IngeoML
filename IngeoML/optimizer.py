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
from itertools import product
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax import nn
import optax
from IngeoML.utils import Batches, balance_class_weigths, progress_bar, error, error_binary


def adam(parameters, batches, objective, 
         epochs: int=5, learning_rate: float=1e-2, 
         every_k_schedule: int=None,
         n_iter_no_change: int=jnp.inf,
         validation=None, model=None,
         validation_score=None, **kwargs):
    """adam optimizer"""

    @jax.jit
    def update_finite(a, b):
        m = jnp.isfinite(b)
        return jnp.where(m, b, a)

    @jax.jit
    def evaluacion(parameters, estado, X, y, weigths):
        grads = objective_grad(parameters, X, y, weigths)
        updates, estado = optimizador.update(grads, estado, parameters)
        parameters = optax.apply_updates(parameters, updates)
        return parameters, estado

    def _validation_score():
        if validation is None:
            return - jnp.inf
        X, y = validation
        hy = model(parameters, X)
        if y.ndim == 1:
            hy = np.where(hy.flatten() > 0, 1, 0)
        else:
            hy = hy.argmax(axis=1)
            y = y.argmax(axis=1)
        return validation_score(y, hy)

    optimizador = optax.adam(learning_rate=learning_rate, **kwargs)
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
    i = 1
    n_iter_no_change = n_iter_no_change * every_k_schedule
    for _, (X, y, weigths) in progress_bar(product(range(epochs),
                                                   batches), total=total):
        p, estado = evaluacion(parameters, estado, X, y, weigths)
        parameters = jax.tree_map(update_finite, parameters, p)
        if (i % every_k_schedule) == 0:
            comp = _validation_score()
            if comp > fit[1]:
                fit = (i, comp, parameters)
        if (i - fit[0]) > n_iter_no_change:
            return fit[-1]
        i += 1
    if validation is None or _validation_score() > fit[1]:
        return parameters
    return fit[-1]


def classifier(parameters, model, X, y, batches=None, array=jnp.array,
               class_weight: str='balanced', n_iter_no_change: int=jnp.inf,
               deviation=None, n_outputs: int=None, validation=None,
               **kwargs):
    """Classifier optimized with optax"""

    @jax.jit
    def deviation_model_binary(params, X, y, weigths):
        hy = model(params, X)
        hy = nn.sigmoid(hy)
        hy = hy.flatten()
        return deviation(y, hy, weigths)

    @jax.jit
    def deviation_model(params, X, y, weigths):
        hy = model(params, X)
        hy = nn.softmax(hy, axis=-1)
        return deviation(y, hy, weigths)
    
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
        batches = Batches() if batches is None else batches
        batches_ = []
        if class_weight == 'balanced':
            splits = batches.split(y=y)
            balance = balance_class_weigths
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
        if n_outputs == 1:
            objective = deviation_model_binary
            if deviation is None:
                deviation = error_binary
        else:
            objective = deviation_model
            if deviation is None:
                deviation = error
        return objective, deviation

    if n_outputs is None:
        n_outputs = model(parameters, array(X[:1])).shape[-1]
    y_enc = encode(y, n_outputs, validation)
    validation, X, y_enc, y = _validation(validation, X, y_enc, y)
    batches_, splits = create_batches(batches)
    if n_iter_no_change < jnp.inf and validation is None:
        jaccard = Batches.jaccard(splits)
        index = jaccard.argmin()
        validation = batches_[index][:2]
        del batches_[index]
    objective, deviation = _objective(deviation)
    return adam(parameters, batches_, objective,
                n_iter_no_change=n_iter_no_change,
                validation=validation, model=model,
                **kwargs)
