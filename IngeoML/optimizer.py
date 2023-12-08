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
from sklearn.preprocessing import OneHotEncoder
import jax
import jax.numpy as jnp
import optax
from IngeoML.utils import Batches



def adam(parameters, batches, objective, 
         epochs: int=5, learning_rate: float=1e-2, 
         every_k_schedule: int=None,
         **kwargs):
    """adam optimizer """

    @jax.jit
    def update_finite(a, b):
        m = jnp.isfinite(b)
        return jnp.where(m, b, a)

    @jax.jit
    def evaluacion(parameters, estado, X, y):
        grads = objective_grad(parameters, X, y)
        updates, estado = optimizador.update(grads, estado, parameters)
        parameters = optax.apply_updates(parameters, updates)
        return parameters, estado

    optimizador = optax.adam(learning_rate=learning_rate, **kwargs)
    _ = every_k_schedule if every_k_schedule is not None else len(batches)
    optimizador = optax.MultiSteps(optimizador, every_k_schedule=_)
    estado = optimizador.init(parameters)
    objective_grad  = jax.grad(objective)
    for _ in range(epochs):
        for X, y in batches:
            p, estado = evaluacion(parameters, estado, X, y)
            parameters = jax.tree_map(update_finite, parameters, p)
    return parameters


def classifier(parameters, model, X, y,
               batch_size: int=64, array=jnp.array,
               **kwargs):
    """Classifier optimized with optax"""

    @jax.jit
    def media_entropia_cruzada(params, X, y):
        hy = model(params, X)
        hy = jax.nn.softmax(hy, axis=0)
        return - ((y * jnp.log(hy)).sum(axis=1) * pesos).sum()

    encoder = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    y_enc = encoder.transform(y.reshape(-1, 1))
    batches = Batches(size=batch_size)
    batches = [(array(X[idx]), array(y_enc[idx]))
               for idx in batches.split(y=y)]
    pesos = jnp.ones(batches[0][0].shape[0])
    return adam(parameters, batches, media_entropia_cruzada, **kwargs)
