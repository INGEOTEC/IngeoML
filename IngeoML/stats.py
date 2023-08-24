from sklearn.metrics import f1_score
from typing import Callable
import numpy as np


class BootstrapSample(object):
    def __init__(self, N: int, num_samples: int=500):
        self.N = N
        self.num_samples = num_samples

    @property
    def N(self):
        return self._N 
    
    @N.setter
    def N(self, value):
        self._N = value

    @property
    def num_samples(self):
        return self._num_samples
    
    @num_samples.setter
    def num_samples(self, value):
        self._num_samples = value

    @property
    def samples(self):
        try:
            return self._samples
        except AttributeError:
            _ = np.random.randint(self.N, size=(self.num_samples, self.N))
            self._samples = _
            return _
        

class CI(object):
    def __init__(self, populations: dict={}, 
                 statistic: Callable[[np.ndarray], float]=np.mean,
                 alpha: float=0.05,
                 num_samples: int=500) -> None:
        self.populations = populations
        for i in populations.values():
            dim = i.shape[0]
            break
        self._bootstrap = BootstrapSample(dim,
                                          num_samples=num_samples)
        self.statistic = statistic
        self.alpha = alpha

    @property
    def bootstrap(self):
        return self._bootstrap.samples

    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def statistic(self):
        return self._statistic
    
    @statistic.setter
    def statistic(self, value):
        self._statistic = value

    @property
    def populations(self):
        return self._pops
    
    @property
    def statistic_samples(self):
        try:
            return self._stat_samples
        except AttributeError:
            self._stat_samples = dict()
            return self._stat_samples

    @populations.setter
    def populations(self, value):
        self._pops = value

    def samples(self, key):
        if key in self.statistic_samples:
            return self.statistic_samples[key]
        data = self.populations[key]
        output = np.array([self.statistic(data[s])
                           for s in self.bootstrap])
        self.statistic_samples[key] = output
        return output

    def confidence_interval(self, key):
        B = self.samples(key)
        alpha = self.alpha
        return (np.percentile(B, alpha * 100), 
                np.percentile(B, (1 - alpha) * 100))


class Difference(CI):
    def __init__(self, y: np.ndarray, 
                 algorithms: dict={}, 
                 performance: Callable[[np.ndarray, np.ndarray], float]=lambda y, hy: f1_score(y, hy, average='macro'),
                 **kwargs) -> None:
        super(Difference, self).__init__(populations=algorithms, statistic=performance)
        self.y = y
        self._dist = dict()
        self._delta = dict()
        self._pvalue_r = dict()
        self._pvalue_l = dict()

    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        self._y = value

    @property
    def best(self):
        try:
            return self._best
        except AttributeError:
            y = self.y
            best = (None, -np.inf)
            for k, v in self.populations.items():
                perf = self.statistic(y, v)
                if perf > best[1]:
                    best = (k, perf)
            self._best = best[0]
            return self._best

    def delta(self, key):
        assert key != self.best
        if key in self._delta:
            return self._delta[key]
        y = self.y
        algs = self.populations
        perf = self.statistic
        delta = perf(y, algs[self.best]) - perf(y, algs[key])
        self._delta[key] = delta
        return delta
    
    def samples(self, key):
        if key in self.statistic_samples:
            return self.statistic_samples[key]
        data = self.populations[key]
        y = self.y
        output = np.array([self.statistic(y[s], data[s])
                           for s in self.bootstrap])
        self.statistic_samples[key] = output
        return output    
    
    @property
    def best_performance(self):
        return self.samples(self.best)
        
    def distribution(self, key):
        best = self.best
        assert key != best
        if key in self._dist:
            return self._dist[key]
        output = self.best_performance - self.samples(key)
        self._dist[key] = output
        return output

    def pvalue(self, key, side='right'):
        assert side in ['left', 'right']
        assert key != self.best
        if side == 'right':
            if key in self._pvalue_r:
                return self._pvalue_r[key]
        elif key in self._pvalue_l:
            return self._pvalue_l[key]
        c = 0
        delta_2 = 2 * self.delta(key)
        delta_i = self.distribution(key)
        if side == 'right':
            c = (delta_i >= delta_2).mean()
        else:
            c = (delta_i < 0).mean()
        if side == 'right':
            self._pvalue_r[key] = c
        else:
            self._pvalue_l[key] = c
        return c
    
    def sort(self, side='right'):
        best = self.best
        algs = [(k, self.pvalue(k, side=side))
                for k in self.populations if k != best]
        algs.sort(key=lambda x: x[1], reverse=True)
        return [k for k, _ in algs]
                