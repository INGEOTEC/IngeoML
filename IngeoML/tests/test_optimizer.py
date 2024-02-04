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
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import jax.numpy as jnp
from jax import nn
import jax
from IngeoML.optimizer import optimize, classifier, regression
from IngeoML.utils import Batches, cross_entropy, soft_error, soft_comp_macro_f1, soft_f1_score


def test_optimize():
    """Test optimize"""

    @jax.jit
    def modelo(params, X):
        Y = X @ params['W'] + params['W0']
        return Y

    @jax.jit
    def media_entropia_cruzada(params, X, y, pesos):
        hy = modelo(params, X)
        hy = jax.nn.softmax(jnp.array(hy), axis=1)
        return - ((y * jnp.log(hy)).sum(axis=1) * pesos).sum()

    X, y = load_iris(return_X_y=True)
    m = LinearSVC(dual='auto').fit(X, y)
    parameters = dict(W=m.coef_.T,
                      W0=m.intercept_)
    encoder = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    batches = Batches()
    a = jnp.array
    y_enc = encoder.transform(y.reshape(-1, 1))
    batches = [[a(X[idx]), a(y_enc[idx])]
               for idx in batches.split(y=y)]
    pesos = jnp.ones(batches[0][0].shape[0])
    for b in batches:
        b.append(pesos)
    p = optimize(parameters, batches, media_entropia_cruzada)
    assert np.fabs(p['W'] - parameters['W']).sum() > 0
    fit1 = media_entropia_cruzada(parameters, *batches[0])
    fit2 = media_entropia_cruzada(p, *batches[0])    
    assert fit2 < fit1


def test_optimize_model_args():
    """Test optimize"""

    @jax.jit
    def modelo(params, X, X2):
        Y = X2 @ params['W'] + params['W0']
        return Y

    @jax.jit
    def media_entropia_cruzada(params, X, y, pesos, *model_args):
        hy = modelo(params, X, *model_args)
        hy = jax.nn.softmax(jnp.array(hy), axis=1)
        return - ((y * jnp.log(hy)).sum(axis=1) * pesos).sum()

    X, y = load_iris(return_X_y=True)
    m = LinearSVC(dual='auto').fit(X, y)
    parameters = dict(W=m.coef_.T,
                      W0=m.intercept_)
    encoder = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    batches = Batches()
    a = jnp.array
    y_enc = encoder.transform(y.reshape(-1, 1))
    batches = [[a(X[idx]), a(y_enc[idx]),
                jnp.ones(idx.shape[0]), a(X[idx])]
               for idx in batches.split(y=y)]
    p = optimize(parameters, batches, media_entropia_cruzada)
    assert np.fabs(p['W'] - parameters['W']).sum() > 0
    fit1 = media_entropia_cruzada(parameters, *batches[0])
    fit2 = media_entropia_cruzada(p, *batches[0])
    assert fit2 < fit1    


def test_classifier():
    """Classifier optimize with jax"""
    from sklearn.metrics import recall_score
    from sklearn.datasets import load_wine

    @jax.jit
    def modelo(params, X):
        Y = X @ params['W'] + params['W0']
        return Y

    X, y = load_wine(return_X_y=True)
    st = StratifiedShuffleSplit(n_splits=1, train_size=10,
                                random_state=0)
    tr, _ = next(st.split(X, y))    
    m = LinearSVC(dual='auto').fit(X[tr], y[tr])
    parameters = dict(W=jnp.array(m.coef_.T),
                      W0=jnp.array(m.intercept_))
    p, evol = classifier(parameters, modelo, X, y,
                         # learning_rate=1e-3,
                         n_iter_no_change=10,
                         return_evolution=True)
    evol = np.array([x[1] for x in evol])
    assert np.any(np.diff(evol) != 0)
    X, y = load_breast_cancer(return_X_y=True)
    index = np.arange(X.shape[0])
    np.random.shuffle(index)    
    m = LinearSVC(dual='auto').fit(X[index[:400]], y[index[:400]])
    parameters = dict(W=jnp.array(m.coef_.T),
                      W0=jnp.array(m.intercept_))
    p2, evol = classifier(parameters, modelo, X, y,
                          n_iter_no_change=10,
                          return_evolution=True)
    evol = np.array([x[1] for x in evol])
    assert np.any(np.diff(evol) != 0)


def test_classifier_model_args():
    """Classifier optimize with jax"""
    from sklearn.metrics import recall_score
    from sklearn.datasets import load_wine

    @jax.jit
    def modelo(params, X, X2):
        Y = X2 @ params['W'] + params['W0']
        return Y

    X, y = load_wine(return_X_y=True)
    st = StratifiedShuffleSplit(n_splits=1, train_size=10,
                                random_state=0)
    tr, _ = next(st.split(X, y))
    m = LinearSVC(dual='auto').fit(X[tr], y[tr])
    parameters = dict(W=jnp.array(m.coef_.T),
                      W0=jnp.array(m.intercept_))
    p, evol = classifier(parameters, modelo, X, y,
                         # learning_rate=1e-3,
                         n_iter_no_change=10,
                         return_evolution=True,
                         model_args=(X,))
    evol = np.array([x[1] for x in evol])
    assert np.any(np.diff(evol) != 0)


def test_classifier_callable_parameter():
    """Classifier optimize with jax"""
    from sklearn.metrics import recall_score
    from sklearn.datasets import load_wine

    @jax.jit
    def modelo(params, X, X2):
        Y = X2 @ params['W'] + params['W0']
        return Y
    
    def initial_parameters(X, y, X2):
        y = y.argmax(axis=1)
        st = StratifiedShuffleSplit(n_splits=1, train_size=10,
                                    random_state=0)
        tr, _ = next(st.split(X2, y))
        m = LinearSVC(dual='auto').fit(X2[tr], y[tr])
        parameters = dict(W=jnp.array(m.coef_.T),
                          W0=jnp.array(m.intercept_))
        return parameters

    X, y = load_wine(return_X_y=True)
    p, evol = classifier(initial_parameters, modelo, X, y,
                         return_evolution=True,
                         n_iter_no_change=10,
                         model_args=(X,))
    evol = np.array([x[1] for x in evol])
    assert np.any(np.diff(evol) != 0)


def test_regression():
    """Test estimator using a regression"""
    @jax.jit
    def modelo(params, X):
        Y = X @ params['W'] + params['W0']
        return nn.sigmoid(Y).flatten()

    X, y = load_breast_cancer(return_X_y=True)
    m = LinearRegression().fit(X, y)
    parameters = dict(W=jnp.array(m.coef_.T),
                      W0=jnp.array(m.intercept_))
    
    p2, evol = regression(parameters, modelo, X, y,
                          return_evolution=True)         
    assert len(evol) > 5
    diff = p2['W0'] - parameters['W0']
    assert np.fabs(diff).sum() > 0


def test_classifier_early_stopping():
    """Test early stopping"""

    @jax.jit
    def modelo(params, X):
        Y = X @ params['W'] + params['W0']
        return Y    

    X, y = load_iris(return_X_y=True)
    m = LinearSVC(dual='auto').fit(X, y)
    parameters = dict(W=jnp.array(m.coef_.T),
                      W0=jnp.array(m.intercept_))
    batches = Batches(size=45)
    p = classifier(parameters, modelo, X, y,
                   epochs=10,
                   batches=batches,
                   n_iter_no_change=2,
                   every_k_schedule=2,
                   learning_rate=1e-1)
    X, y = load_breast_cancer(return_X_y=True)
    m = LinearSVC(dual='auto').fit(X, y)
    parameters = dict(W=jnp.array(m.coef_.T),
                      W0=jnp.array(m.intercept_))
    p2 = classifier(parameters, modelo, X, y,
                    epochs=10,
                    batches=batches,
                    n_iter_no_change=2,
                    every_k_schedule=2,
                    learning_rate=1e-1)
    

def test_regression2():
    """Test regression"""
    from scipy.stats import pearsonr
    from sklearn.linear_model import LinearRegression

    def validation_score(y, hy):
        cnt = (hy == 0).sum() + (hy == 1).sum()
        assert cnt < hy.shape[0]
        return 1 / - jnp.nansum(y * jnp.log(hy))

    @jax.jit
    def modelo(params, X):
        Y = X @ params['W'] + params['W0']
        return nn.sigmoid(Y).flatten()
    
    def objective(params, X, y, weights):
        hy = modelo(params, X)
        return - jnp.nansum(y * jnp.log(hy))

    X, y = load_breast_cancer(return_X_y=True)
    m = LinearRegression().fit(X, y)
    params = dict(W=jnp.array(m.coef_.T),
                  W0=jnp.array(m.intercept_))
    batches = Batches()
    a = jnp.array
    batches = [[a(X[idx]), a(y[idx])]
               for idx in batches.split(X)]
    pesos = jnp.ones(batches[0][0].shape[0])
    for b in batches:
        b.append(pesos)
    hy = modelo(params, X)
    pre = 1 / - jnp.nansum(y * jnp.log(hy))
    p, evol = optimize(params, batches[1:], objective,
                       validation=batches[0][:2],
                       every_k_schedule=3,
                       epochs=50,
                       learning_rate=1e-3,
                       n_iter_no_change=5,
                       validation_score=validation_score,
                       discretize_val=False,
                       return_evolution=True,
                       model=modelo)    

    hy = modelo(p, X)
    despues = 1 / - jnp.nansum(y * jnp.log(hy))
    assert despues > pre
    # assert evol is None


def test_classifier_deviation():
    """Test classifier deviation"""
    @jax.jit
    def modelo(params, X):
        Y = X @ params['W'] + params['W0']
        return Y
    
    X, y = load_iris(return_X_y=True)
    m = LinearSVC(dual='auto').fit(X, y)
    parameters = dict(W=m.coef_.T,
                      W0=m.intercept_)

    p = classifier(parameters, modelo, X, y,
                   epochs=1,
                   every_k_schedule=1,
                   deviation=cross_entropy)
    

def test_classifier_error():
    """Test classifier error"""
    @jax.jit
    def modelo(params, X):
        Y = X @ params['W'] + params['W0']
        return Y
    
    X, y = load_iris(return_X_y=True)
    m = LinearSVC(dual='auto').fit(X, y)
    parameters = dict(W=m.coef_.T,
                      W0=m.intercept_)
    # assert modelo(parameters, X).shape[-1] is None
    p = classifier(parameters, modelo, X, y,
                   epochs=1,
                   every_k_schedule=1,
                   deviation=soft_error) 


def test_classifier_validation():
    @jax.jit
    def modelo(params, X):
        Y = X @ params['W'] + params['W0']
        return Y
    
    X, y = load_iris(return_X_y=True)
    split = StratifiedShuffleSplit(n_splits=1,
                                   test_size=0.2)
    m = LinearSVC(dual='auto').fit(X, y)
    parameters = dict(W=m.coef_.T,
                      W0=m.intercept_)
    p = classifier(parameters, modelo, X, y,
                   epochs=3, every_k_schedule=2,
                   n_iter_no_change=2, validation=split)
    
    tr, vs = next(split.split(X, y))
    validation = [X[vs], y[vs]]
    p = classifier(parameters, modelo, X[tr], y[tr],
                   epochs=3, every_k_schedule=2,
                   n_iter_no_change=2,
                   validation=validation)


def test_classifier_evolution():
    """Test the evolution feature"""
    @jax.jit
    def modelo(params, X):
        Y = X @ params['W'] + params['W0']
        return Y

    X, y = load_iris(return_X_y=True)
    m = LinearSVC(dual='auto').fit(X, y)
    parameters = dict(W=jnp.array(m.coef_.T),
                      W0=jnp.array(m.intercept_))
    p, evolution = classifier(parameters, modelo, X, y,
                              return_evolution=True,
                              n_iter_no_change=2,
                              deviation=soft_comp_macro_f1)
    assert len(evolution) and evolution[0][1] > 0.85


def test_classifier_discretize_val():
    """Classifier optimize with jax"""
    from sklearn.metrics import recall_score
    from sklearn.datasets import load_wine

    @jax.jit
    def soft_macro_f1(y, hy, w=None):
        return soft_f1_score(y, nn.softmax(hy, axis=-1)).mean()

    @jax.jit
    def modelo(params, X, X2):
        Y = X2 @ params['W'] + params['W0']
        return Y
    
    def initial_parameters(X, y, X2):
        y = y.argmax(axis=1)
        st = StratifiedShuffleSplit(n_splits=1, train_size=10,
                                    random_state=0)
        tr, _ = next(st.split(X2, y))
        m = LinearSVC(dual='auto').fit(X2[tr], y[tr])
        parameters = dict(W=jnp.array(m.coef_.T),
                          W0=jnp.array(m.intercept_))
        return parameters

    X, y = load_wine(return_X_y=True)
    p, evol = classifier(initial_parameters, modelo, X, y,
                         return_evolution=True,
                         discretize_val=False,
                         validation_score=soft_macro_f1,
                         model_args=(X,))
    evol = np.array([x[1] for x in evol])
    assert np.any(np.diff(evol) != 0)


def test_classifier_validation_zero():
    """Classifier optimize with jax"""
    from sklearn.metrics import recall_score
    from sklearn.datasets import load_wine

    @jax.jit
    def modelo(params, X, X2):
        Y = X2 @ params['W'] + params['W0']
        return Y
    
    def initial_parameters(X, y, X2):
        y = y.argmax(axis=1)
        st = StratifiedShuffleSplit(n_splits=1, train_size=20,
                                    random_state=0)
        tr, _ = next(st.split(X2, y))
        m = LinearSVC(dual='auto').fit(X2[tr], y[tr])
        parameters = dict(W=jnp.array(m.coef_.T),
                          W0=jnp.array(m.intercept_))
        return parameters

    X, y = load_wine(return_X_y=True)
    p, evol = classifier(initial_parameters, modelo, X, y,
                         return_evolution=True,
                         validation=0,
                         model_args=(X,))
    assert len(evol) > 8


def test_classifier_distribution():
    """Classifier optimize with jax"""
    from sklearn.metrics import recall_score
    from sklearn.datasets import load_wine

    @jax.jit
    def modelo(params, X, X2):
        Y = X2 @ params['W'] + params['W0']
        return nn.softmax(Y, axis=1)
    
    def initial_parameters(X, y, X2):
        y = y.argmax(axis=1)
        st = StratifiedShuffleSplit(n_splits=1, train_size=20,
                                    random_state=0)
        tr, _ = next(st.split(X2, y))
        m = LinearSVC(dual='auto').fit(X2[tr], y[tr])
        parameters = dict(W=jnp.array(m.coef_.T),
                          W0=jnp.array(m.intercept_))
        return parameters

    X, y = load_wine(return_X_y=True)
    p, evol = classifier(initial_parameters, modelo, X, y,
                         return_evolution=True,
                         validation=0, distribution=True,
                         model_args=(X,))
    assert len(evol) > 8    