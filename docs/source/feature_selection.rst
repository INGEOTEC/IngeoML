.. _feature_selection:

====================================
Feature Selection
====================================
.. image:: https://github.com/INGEOTEC/IngeoML/actions/workflows/test.yaml/badge.svg
		:target: https://github.com/INGEOTEC/IngeoML/actions/workflows/test.yaml

.. image:: https://coveralls.io/repos/github/INGEOTEC/IngeoML/badge.svg?branch=develop
		:target: https://coveralls.io/github/INGEOTEC/IngeoML?branch=develop

.. image:: https://badge.fury.io/py/IngeoML.svg
		:target: https://badge.fury.io/py/IngeoML

.. image:: https://readthedocs.org/projects/ingeoml/badge/?version=latest
    :target: https://ingeoml.readthedocs.io/en/latest/?badge=latest


:py:class:`IngeoML.feature_selection` API
--------------------------------------------

.. autoclass:: IngeoML.feature_selection.SelectFromLinearSVC
	:members: fit, transform, features

.. autoclass:: IngeoML.feature_selection.SelectFromModelCV
   :members: fit, max_iter, cv, scoring, min_features_to_select