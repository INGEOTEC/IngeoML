from BPerfAnalysis.base import BootstrapSample, CI, Difference
import numpy as np


def problem_algorithms():
    labels = [0, 0, 0, 0, 0,
              1, 1, 1, 1, 1]
    a = [0, 0, 0, 0, 0,
         1, 1, 1, 1, 0]
    b = [0, 0, 1, 0, 0,
         1, 1, 1, 1, 0]
    c = [0, 0, 0, 1, 0,
         1, 1, 0, 1, 0]
    return (np.array(labels),
            dict(a=np.array(a),
                 b=np.array(b),
                 c=np.array(c)))


def test_BootstrapSample():
    sample = BootstrapSample(N=10, num_samples=26)
    s = sample.samples
    assert s.shape == (26, 10)
    assert np.all(s == sample.samples)


def test_CI():
    labels, algs = problem_algorithms()
    ci = CI(populations=algs)
    a = ci.confidence_interval('a')
    assert len(a) == 2 and a[0] > 0 and a[1] < 0.8


def test_Difference_ci():
    labels, algs = problem_algorithms()
    diff = Difference(labels, algs)
    a = diff.confidence_interval('a')
    assert a[0] > 0.6 and a[1] <= 1.0


def test_Difference_best():
    labels, algs = problem_algorithms()
    diff = Difference(labels, algs)
    assert diff.best == 'a'


def test_Difference_delta():
    labels, algs = problem_algorithms()
    diff = Difference(labels, algs)
    assert diff.delta('b') > 0 and diff.delta('c') > 0


def test_Difference():
    labels, algs = problem_algorithms()
    diff = Difference(labels, algs)
    assert diff.best == 'a'
    assert diff.pvalue('b') > diff.pvalue('c')


def test_Difference_sort():
    labels, algs = problem_algorithms()
    diff = Difference(labels, algs)
    for x, r in zip(diff.sort(), ['b', 'c']):
        assert x == r