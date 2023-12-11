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
import jax.lax as lax
import optax
from IngeoML.utils import Batches, balance_class_weigths, progress_bar


def adam(parameters, batches, objective, 
         epochs: int=5, learning_rate: float=1e-2, 
         every_k_schedule: int=None,
         early_stopping: int=None,
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

    optimizador = optax.adam(learning_rate=learning_rate, **kwargs)
    if validation_score is None:
        validation_score = lambda y, hy: f1_score(y, hy, average='macro')
    _ = every_k_schedule if every_k_schedule is not None else len(batches)
    every_k_schedule = _
    optimizador = optax.MultiSteps(optimizador,
                                   every_k_schedule=every_k_schedule)
    estado = optimizador.init(parameters)
    objective_grad  = jax.grad(objective)
    total = epochs * len(batches)
    fit, best = None, None
    i = 0        
    for _, (X, y, weigths) in progress_bar(product(range(epochs),
                                                   batches), total=total):
        p, estado = evaluacion(parameters, estado, X, y, weigths)
        parameters = jax.tree_map(update_finite, parameters, p)
        if validation is not None and (i % every_k_schedule) == 0:
            X, y, weigths = validation
            hy = model(parameters, X)
            if y.ndim == 1:
                hy = np.where(hy.flatten() > 0, 1, 0)
            else:
                hy = hy.argmax(axis=1)
                y = y.argmax(axis=1)
            comp = validation_score(y, hy)
            if fit is None or comp > fit[1]:
                fit = (i, comp)
                best = parameters
            elif (i - fit[0]) // every_k_schedule >= early_stopping:
                return best
        i += 1
    return parameters


def classifier(parameters, model, X, y,
               batches=None, array=jnp.array,
               class_weight: str='balanced',
               early_stopping: int=None,
               **kwargs):
    """Classifier optimized with optax"""

    @jax.jit
    def suma_entropia_cruzada(params, X, y, weigths):
        hy = model(params, X)
        hy = jax.nn.softmax(hy, axis=0)
        return - ((y * jnp.log(hy)).sum(axis=1) * weigths).sum()

    @jax.jit
    def entropia_cruzada(y, hy):
        _ = lax.cond(y == 1, lambda w: jnp.log(w), lambda w: jnp.log(1 - w), hy)
        return lax.cond(_ == -jnp.inf, lambda w: jnp.log(1e-6), lambda w: w, _)

    @jax.jit
    def media_entropia_cruzada_binaria(params, X, y, weigths):
        hy = model(params, X)
        hy = 1 / (1 + jnp.exp(-hy))
        hy = hy.flatten()
        return - lax.fori_loop(0, y.shape[0],
                               lambda i, x: x + weigths[i] * entropia_cruzada(y[i], hy[i]),
                               1) / y.shape[0]

    validation = None
    batches = Batches() if batches is None else batches
    labels = np.unique(y)
    if labels.shape[0] == 2:
        h = {v:k for k, v in enumerate(labels)}
        y_enc = np.array([h[x] for x in y])
    else:
        encoder = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
        y_enc = encoder.transform(y.reshape(-1, 1))
    batches_ = []
    if class_weight == 'balanced':
        splits = batches.split(y=y)
        for idx in splits:
            batches_.append((array(X[idx]),
                             jnp.array(y_enc[idx]),
                             jnp.array(balance_class_weigths(y[idx]))))
    else:
        splits = batches.split(y=y)
        for idx in splits:
            batches_.append((array(X[idx]),
                             jnp.array(y_enc[idx]),
                             jnp.ones(idx.shape[0])))
    if early_stopping is not None:
        jaccard = batches.jaccard(splits)
        index = jaccard.argmin()
        validation = batches_[index]
        del batches_[index]
    if labels.shape[0] == 2:
        objective = media_entropia_cruzada_binaria
    else:
        objective = suma_entropia_cruzada
    return adam(parameters, batches_,
                objective,
                early_stopping=early_stopping,
                validation=validation, model=model,
                **kwargs)
