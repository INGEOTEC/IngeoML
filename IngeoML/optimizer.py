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
from IngeoML.utils import Batches, balance_class_weigths, progress_bar, cross_entropy


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
               deviation=cross_entropy,
               n_outputs: int=None,
               **kwargs):
    """Classifier optimized with optax"""

    @jax.jit
    def deviation_model_binary(params, X, y, weigths):
        hy = model(params, X)
        hy = 1 / (1 + jnp.exp(-hy))
        hy = hy.flatten()
        return deviation(y, hy, weigths)

    @jax.jit
    def deviation_model(params, X, y, weigths):
        hy = model(params, X)
        hy = jax.nn.softmax(hy, axis=-1)
        return deviation(y, hy, weigths)
    
    def encode(y, n_outputs):
        if n_outputs == 1:
            labels = np.unique(y)            
            h = {v:k for k, v in enumerate(labels)}
            y_enc = np.array([h[x] for x in y])
        else:
            encoder = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
            y_enc = encoder.transform(y.reshape(-1, 1))
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

    if n_outputs is None:
        n_outputs = model(parameters, X).shape[-1]
    y_enc = encode(y, n_outputs)
    batches_, splits = create_batches(batches)
    validation = None
    if early_stopping is not None:
        jaccard = Batches.jaccard(splits)
        index = jaccard.argmin()
        validation = batches_[index]
        del batches_[index]

    if n_outputs == 1:
        objective = deviation_model_binary
    else:
        objective = deviation_model
    return adam(parameters, batches_, objective,
                early_stopping=early_stopping,
                validation=validation, model=model,
                **kwargs)
