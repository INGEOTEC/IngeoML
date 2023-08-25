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
from sklearn.metrics import f1_score
from typing import Any, Callable
import numpy as np


class StatisticSamples(object):
    """Apply the statistic to `num_samples` samples taken with replacement from the population (arguments).

    :param statistic: Statistic
    :type statistic: Callable
    :param num_samples: Number of bootstrap samples, default=500
    :type num_samples: int

    >>> from IngeoML.bootstrap import StatisticSamples
    >>> from sklearn.metrics import accuracy_score
    >>> import numpy as np
    >>> statistic = StatisticSamples(num_samples=10, statistic=np.mean)
    >>> empirical_distribution = np.r_[[3, 4, 5, 2, 4]]
    >>> statistic(empirical_distribution)
    array([2.8, 3.6, 3.6, 3.6, 2.6, 4. , 2.8, 3. , 3.8, 3.6])
    >>> labels = np.r_[[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
    >>> pred   = np.r_[[0, 0, 1, 0, 0, 1, 1, 1, 0, 1]]
    >>> acc = StatisticSamples(num_samples=15, statistic=accuracy_score)
    >>> acc(labels, pred)
    array([0.9, 0.8, 0.7, 1. , 0.6, 1. , 0.7, 0.9, 0.9, 0.8, 0.9, 0.8, 0.8, 0.8, 0.8])
    """

    def __init__(self,
                 statistic: Callable[[np.ndarray], float]=np.mean,
                 num_samples: int=500):
        self.statistic = statistic
        self.num_samples = num_samples

    @property
    def statistic(self):
        """Statistic"""
        return self._statistic
    
    @statistic.setter
    def statistic(self, value):
        self._statistic = value

    @property
    def num_samples(self):
        """Number of bootstrap samples"""
        return self._num_samples
    
    @num_samples.setter
    def num_samples(self, value):
        self._num_samples = value

    def samples(self, N):
        """Samples
        
        :param N: Population size
        :type N: int
        """
        def inner(N):
            _ = np.random.randint(N, size=(self.num_samples, N))
            self._samples = _
            return self._samples
        try:
            if self._samples.shape[0] == N:
                return self._samples
            else:
                return inner(N)
        except AttributeError:
            return inner(N)
        
    def __call__(self, *args: np.ndarray) -> np.ndarray:
        """Population where the bootstrap process will be performed. 

        :param *args: Population
        :type *args: np.ndarray
        """
        B = []
        statistic = self.statistic
        for s in self.samples(args[0].shape[0]):
            _ = [arg[s] for arg in args]
            B.append(statistic(*_))
        return np.array(B)
        

# class CI(object):
#     """Confidence Intervals
    
#     :param populations: 
#     :type populations: dict
#     """
#     def __init__(self, populations: dict={}, 
#                  statistic: Callable[[np.ndarray], float]=np.mean,
#                  alpha: float=0.05,
#                  num_samples: int=500) -> None:
#         self.populations = populations
#         for i in populations.values():
#             dim = i.shape[0]
#             break
#         self._bootstrap = BootstrapSample(dim,
#                                           num_samples=num_samples)
#         self.statistic = statistic
#         self.alpha = alpha

#     @property
#     def bootstrap(self):
#         return self._bootstrap.samples

#     @property
#     def alpha(self):
#         return self._alpha
    
#     @alpha.setter
#     def alpha(self, value):
#         self._alpha = value

#     @property
#     def statistic(self):
#         return self._statistic
    
#     @statistic.setter
#     def statistic(self, value):
#         self._statistic = value

#     @property
#     def populations(self):
#         return self._pops
    
#     @property
#     def statistic_samples(self):
#         try:
#             return self._stat_samples
#         except AttributeError:
#             self._stat_samples = dict()
#             return self._stat_samples

#     @populations.setter
#     def populations(self, value):
#         self._pops = value

#     def samples(self, key):
#         if key in self.statistic_samples:
#             return self.statistic_samples[key]
#         data = self.populations[key]
#         output = np.array([self.statistic(data[s])
#                            for s in self.bootstrap])
#         self.statistic_samples[key] = output
#         return output

#     def confidence_interval(self, key):
#         B = self.samples(key)
#         alpha = self.alpha
#         return (np.percentile(B, alpha * 100), 
#                 np.percentile(B, (1 - alpha) * 100))


# class Difference(CI):
#     def __init__(self, y: np.ndarray, 
#                  algorithms: dict={}, 
#                  performance: Callable[[np.ndarray, np.ndarray], float]=lambda y, hy: f1_score(y, hy, average='macro'),
#                  **kwargs) -> None:
#         super(Difference, self).__init__(populations=algorithms, statistic=performance)
#         self.y = y
#         self._dist = dict()
#         self._delta = dict()
#         self._pvalue_r = dict()
#         self._pvalue_l = dict()

#     @property
#     def y(self):
#         return self._y
    
#     @y.setter
#     def y(self, value):
#         self._y = value

#     @property
#     def best(self):
#         try:
#             return self._best
#         except AttributeError:
#             y = self.y
#             best = (None, -np.inf)
#             for k, v in self.populations.items():
#                 perf = self.statistic(y, v)
#                 if perf > best[1]:
#                     best = (k, perf)
#             self._best = best[0]
#             return self._best

#     def delta(self, key):
#         assert key != self.best
#         if key in self._delta:
#             return self._delta[key]
#         y = self.y
#         algs = self.populations
#         perf = self.statistic
#         delta = perf(y, algs[self.best]) - perf(y, algs[key])
#         self._delta[key] = delta
#         return delta
    
#     def samples(self, key):
#         if key in self.statistic_samples:
#             return self.statistic_samples[key]
#         data = self.populations[key]
#         y = self.y
#         output = np.array([self.statistic(y[s], data[s])
#                            for s in self.bootstrap])
#         self.statistic_samples[key] = output
#         return output    
    
#     @property
#     def best_performance(self):
#         return self.samples(self.best)
        
#     def distribution(self, key):
#         best = self.best
#         assert key != best
#         if key in self._dist:
#             return self._dist[key]
#         output = self.best_performance - self.samples(key)
#         self._dist[key] = output
#         return output

#     def pvalue(self, key, side='right'):
#         assert side in ['left', 'right']
#         assert key != self.best
#         if side == 'right':
#             if key in self._pvalue_r:
#                 return self._pvalue_r[key]
#         elif key in self._pvalue_l:
#             return self._pvalue_l[key]
#         c = 0
#         delta_2 = 2 * self.delta(key)
#         delta_i = self.distribution(key)
#         if side == 'right':
#             c = (delta_i >= delta_2).mean()
#         else:
#             c = (delta_i < 0).mean()
#         if side == 'right':
#             self._pvalue_r[key] = c
#         else:
#             self._pvalue_l[key] = c
#         return c
    
#     def sort(self, side='right'):
#         best = self.best
#         algs = [(k, self.pvalue(k, side=side))
#                 for k in self.populations if k != best]
#         algs.sort(key=lambda x: x[1], reverse=True)
#         return [k for k, _ in algs]
                