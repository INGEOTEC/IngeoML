.. _utils:

====================================
Utils
====================================
.. image:: https://github.com/INGEOTEC/IngeoML/actions/workflows/test.yaml/badge.svg
		:target: https://github.com/INGEOTEC/IngeoML/actions/workflows/test.yaml

.. image:: https://coveralls.io/repos/github/INGEOTEC/IngeoML/badge.svg?branch=develop
		:target: https://coveralls.io/github/INGEOTEC/IngeoML?branch=develop

.. image:: https://badge.fury.io/py/IngeoML.svg
		:target: https://badge.fury.io/py/IngeoML

.. image:: https://readthedocs.org/projects/ingeoml/badge/?version=latest
    :target: https://ingeoml.readthedocs.io/en/latest/?badge=latest


:py:class:`IngeoML.utils` API
--------------------------------------------

.. autoclass:: IngeoML.utils.Batches
    :members: split, blocks, distribution

.. automodule:: IngeoML.utils
    :members: balance_class_weights, cross_entropy, soft_error, soft_BER, soft_recall, soft_precision, soft_f1_score, soft_comp_macro_f1, cos_similarity, cos_distance